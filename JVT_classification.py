import sys
import os
import torch
from tqdm import tqdm
import h5py
import torch.nn as nn
import numpy as np
from torchvision.ops import MLP
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pytz

##################################################################################################################
## Some general functions ########################################################################################
##################################################################################################################


def main():

    # same as in TMAE.py
    def create_dataloader(folder_path ,batch_size, shuffle=False):
        dataset = H5Dataset(folder_path,batch_size)
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers = int(sys.argv[2]), prefetch_factor=8)

    ##################################################################################################################
    ## Class declarations ############################################################################################
    ##################################################################################################################

    # same as in TMAE.py
    class TransformerSetEncoder(nn.Module):
        def __init__(self,hidden_dim, num_layers, num_heads):
            super().__init__()
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*2,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        def forward(self, x, mask):
            src_key_padding_mask = ~mask.bool()  # True = ignore
            return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    # same as in TMAE.py
    class TMAE(nn.Module):
        def __init__(self,features_number, latent_space_dim ,nlayer,nhead): # doesn not depend on tokens number
            super().__init__()

            self.embedding = nn.Linear(features_number, latent_space_dim).to(device)  
            self.unembedding = nn.Linear(latent_space_dim, features_number).to(device)
            self.encoder = TransformerSetEncoder(hidden_dim=latent_space_dim, num_layers=nlayer, num_heads=nhead).to(device)
            self.decoder = TransformerSetEncoder(hidden_dim=latent_space_dim, num_layers=nlayer, num_heads=nhead).to(device)

            self.extra_token = nn.Parameter(torch.randn(1, 35, latent_space_dim,device=device))  # Extra tokens to be added to the latent space

        def forward(self,x, mask, null_mask):
            x = x[0].to(device)
            mask = mask[0].to(device)
            null_mask = null_mask[0].to(device)
            x_embedded = self.embedding(x)  # Embedding the input features
            latent_space = self.encoder(x_embedded, null_mask)
            # latent_space[~null_mask & mask] = self.extra_token
            replace_mask = (~null_mask & mask)  # [batch, feature_number]
            extra_subset = self.extra_token.expand(latent_space.shape[0], -1, -1)
            latent_space = torch.where(replace_mask.unsqueeze(-1), extra_subset, latent_space)
            decoded = self.decoder(latent_space,mask)
            out = self.unembedding(decoded)  # Unembedding the latent space back to the original feature space
            return out

    # early stopping class to stop training when no improvement is seen after a certain number of epochs 
    class EarlyStopping:
        def __init__(self, patience=5, delta=0, verbose=False):
            self.patience = patience
            self.delta = delta
            self.verbose = verbose
            self.best_loss = None
            self.no_improvement_count = 0
            self.stop_training = False

        def check_early_stop(self, val_loss):
            if self.best_loss is None or val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience:
                    self.stop_training = True
                    if self.verbose:
                        print("Stopping early as no improvement has been observed.")

    # Classification head, that take the latent space from TMAE and output a single value
    # see slides for explanation of the architecture
    class CH(nn.Module) :
        def __init__(self, tokens_num, features_num):
            super().__init__()

            self.tokens_num = tokens_num
            self.features_num = features_num
            layers = []
            layers.append(nn.Linear(features_num, 256))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(256, 128))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(128, 1))
            self.linear = nn.Sequential(*layers)

        def forward(self,x, mask):
            x = x[0].to(device)
            mask = mask[0].to(device)
            mask_unsqueezed = mask.unsqueeze(-1)
            x = x * mask_unsqueezed
            pooled = x.mean(dim=1)
            out = self.linear(pooled)
            return out

    # Combined model to train TMAE and CH together
    class CombinedModel(nn.Module):
        def __init__(self,embedding, encoder, classifier):
            super().__init__()
            self.embedding = embedding
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, x, mask):
            x = self.embedding(x[0])
            encoded = self.encoder(x, mask[0])
            out = self.classifier(encoded, mask)
            return out

    # same as in TMAE.py
    class H5Dataset(Dataset):
        def __init__(self, folder_path, batch_size):
            self.folder_path = folder_path
            self.batch_size = batch_size
            self.file_paths = [os.path.join(folder_path,f) for f in os.listdir(folder_path)]
            self.index_map = []
            for i, file_path in enumerate(self.file_paths):
                with h5py.File(file_path, "r") as f:
                    jets = f["jets"]
                    num_entries = len(jets)
                    for i in range(0, num_entries, batch_size):
                        end = min(i + batch_size, num_entries)
                        self.index_map.append((file_path,i,end))
            self.file_path=self.file_paths[0]
            self.file = h5py.File(self.file_paths[0], "r")
            
        def __len__(self):
            return len(self.index_map)

        def __getitem__(self, idx):
            file_path, start, end = self.index_map[idx]

            if self.file_path!=file_path:
                self.file=h5py.File(file_path, "r")
                self.file_path=file_path

            jets = self.file["jets"]
            batch = jets[start:end]
            masks = self.file["mask"][start:end]==1 # convert to boolean
            labels = self.file["labels"][start:end]

            output ={"x": torch.tensor(batch).float(),"mask": torch.tensor(masks).bool(),"labels": torch.tensor(labels).float()} 

            return output
        
    ###################################################################################################################
    ## Inputs and data ################################################################################################
    ###################################################################################################################


    folder_path = "/eos/user/s/smorand/h5data/v3_cut/01_09"
    folder_path_valid = "/eos/user/s/smorand/h5data/v3_cut_validation/01_09"
    outputs_path = "/eos/user/s/smorand/outputs/"
    # pretrained_model for TMAE encoder
    pretrained_model = "/eos/user/s/smorand/outputs/trained_model/trained_model_T_15616303_2.pth"

    batch_size = 1000
    patience = 5
    delta = 0.0001

    # for naming output files
    job_id = sys.argv[1]
    # whether to use a pretrained TMAE model or not
    pretrained = int(sys.argv[3])
    # number of batches to process in each epoch to try different sample sizes
    limit = int(sys.argv[4])

    out_f1 = f"batch_loss_CH_NFM/batch_loss_{job_id}.csv"
    out_f2 = f"epoch_loss_CH_NFM/epoch_loss_{job_id}.csv"
    out_f3 = f"epoch_loss_CH_NFM/validation_loss_{job_id}.csv"

    ####################################################################################################################
    ## Model and training ##############################################################################################
    ####################################################################################################################
    tz = pytz.timezone("Europe/Zurich")
    
    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print("Creation of Dataloader...")
    loader = create_dataloader(folder_path, batch_size=batch_size, shuffle=False)
    loader_validation = create_dataloader(folder_path_valid, batch_size=batch_size, shuffle=False)

    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print("Initialization of model...")
    model_PT = TMAE(features_number=12, latent_space_dim=128 ,nlayer=2,nhead=4) # TMAE
    if pretrained == 1 :
        print("Using pretrained model...")
        model_PT.load_state_dict(torch.load(pretrained_model, map_location=device))
        out_f1 = f"batch_loss_CH_FM/batch_loss_{job_id}.csv"
        out_f2 = f"epoch_loss_CH_FM/epoch_loss_{job_id}.csv"
        out_f3 = f"epoch_loss_CH_FM/validation_loss_{job_id}.csv"
    embedding = model_PT.embedding.to(device)    
    encoder = model_PT.encoder.to(device)

    # freeze embedding and encoder if using pretrained model
    # for param in embedding.parameters():
    #     param.requires_grad = False
    # for param in encoder.parameters():
    #     param.requires_grad = False

    classifier = CH(35,128).to(device)
    model = CombinedModel(embedding,encoder, classifier).to(device)

    # different learning rates for different parts of the model
    optimizer = torch.optim.Adam([
        {"params": classifier.parameters(), "lr": 1e-3},
        {"params": embedding.parameters(), "lr": 1e-4},        
        {"params": [p for n, p in encoder.named_parameters() if p.requires_grad], "lr": 1e-4}  
    ])

    # sigmoid is directly included in the loss function
    loss_fn = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(f"Early stopping with patience = {patience} and delta = {delta}.")

    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(f"Ready for training.")
    
    n_parameters = 0
    for i in model.parameters():
        n_parameters += i.numel() if i.requires_grad else 0
    print(f"Trainable parameters : {n_parameters}.")

    with (open(outputs_path + out_f1, "w") as f1,
          open(outputs_path + out_f2, "w") as f2,
          open(outputs_path + out_f3, "w") as f3):
        for epoch in range(1000):
            model.train()
            total_loss = 0
            batch_count = 0
            for batch in loader :
                batch_count +=1
                # limit number of batches per epoch to try different sample sizes
                if batch_count > limit :
                    break
                preds = model(batch["x"].to(device),batch["mask"].to(device))
                labels = batch["labels"].squeeze(0).squeeze(-1)
                # create binary labels based on JVT cut at 0.1 and 0.9
                labels_int = torch.where(labels<=0.1, torch.tensor(0.), torch.tensor(1.))
                loss = loss_fn(preds.squeeze(), labels_int.to(device))
                optimizer.zero_grad() # reset gradients
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                f1.write(f"{loss.item()}\n")
            avg_loss = total_loss/batch_count
            print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")
            f2.write(f"{avg_loss}\n")

            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            # all_pred_labels = []
            all_pred = []
            all_labels_int = []
            with torch.no_grad():
                for batch in loader_validation :
                    val_batches += 1
                    preds = model(batch["x"].to(device),batch["mask"].to(device))
                    labels = batch["labels"].squeeze(0).squeeze(-1)
                    labels_int = torch.where(labels<=0.1, torch.tensor(0.), torch.tensor(1.))
                    loss = loss_fn(preds.squeeze(), labels_int.to(device))
                    val_loss += loss.item()       
                    probs = torch.sigmoid(preds.squeeze())             # logits → probs
                    all_pred.append(probs)
                    all_labels_int.append(labels_int)
            all_pred = torch.cat(all_pred).cpu().numpy()
            all_labels_int = torch.cat(all_labels_int)
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('nan')
            # print AUC score and validation loss
            print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | ROC AUC: {roc_auc_score(all_labels_int, all_pred):.4f}")
            f3.write(f"{avg_val_loss}\n")
            early_stopping.check_early_stop(avg_val_loss)
            out_f4 = f"trained_model_CH_FM/trained_model_NFM_{job_id}_{epoch}"
            if pretrained:
                out_f4 = f"trained_model_CH_FM/trained_model_FM_{job_id}_{epoch}"
            torch.save(model.state_dict(),outputs_path + out_f4)
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch + 1}")
                print(f"Best validation loss : {early_stopping.best_loss}")
                break
    ####################################################################################################################

if __name__ == "__main__":
    main()