import sys
import os
import torch
from tqdm import tqdm
import h5py
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from datetime import datetime
import pytz

def main():
    # same as in TMAE.py
    def create_dataloader(folder_path ,batch_size, shuffle=False):
        dataset = H5Dataset(folder_path,batch_size)
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers = int(sys.argv[2]), prefetch_factor=8)

    ##################################################################################################################
    ## Class declarations ############################################################################################
    ##################################################################################################################

    # same as JVT_classification.py
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

    # simple model to test training on scratch
    # this model isn't in slides
    class CH_scratch_0(nn.Module) :
        def __init__(self,features_num):
            super().__init__()

            self.features_num = features_num
            self.linear = nn.Linear(features_num,1)

        def forward(self,x, mask):
            x = x[0].to(device)
            mask = mask[0].to(device)
            mask_unsqueezed = mask.unsqueeze(-1)
            x = x * mask_unsqueezed
            pooled = x.mean(dim=1)
            out = self.linear(pooled)
            return out

    # simple model to test training on scratch, this one match the latent space dimension of the autoencoder
    # useful for comparing the performance of the autoencoder latent space with a simple linear model
    # see slides for details
    class CH_scratch_1(nn.Module) :
        def __init__(self,features_num):
            super().__init__()

            self.features_num = features_num
            self.linear_initial = nn.Sequential(nn.Linear(features_num,128),nn.ReLU())
            self.linear = nn.Linear(128,1)

        def forward(self,x, mask):
            x = x[0].to(device)
            mask = mask[0].to(device)
            mask_unsqueezed = mask.unsqueeze(-1)
            x_layer1 = self.linear_initial(x)
            x_layer1 = x_layer1 * mask_unsqueezed
            pooled = x_layer1.mean(dim=1)
            out = self.linear(pooled)
            return out

    # same as JVT_classification.py
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

    batch_size = 1000
    patience = 200
    delta = 0.0001

    job_id = sys.argv[1]
    scratch_version = int(sys.argv[3])
    limit = int(sys.argv[4])

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
    if scratch_version == 0 :
        print("Using scratch 0 model...")
        model = CH_scratch_0(12)
        out_f1 = f"batch_loss_scratch_0/batch_loss_{job_id}.csv"
        out_f2 = f"epoch_loss_scratch_0/epoch_loss_{job_id}.csv"
        out_f3 = f"epoch_loss_scratch_0/validation_loss_{job_id}.csv"
    elif scratch_version == 1 :
        print("Using scratch 1 model...")
        model = CH_scratch_1(12)
        out_f1 = f"batch_loss_scratch_1/batch_loss_{job_id}.csv"
        out_f2 = f"epoch_loss_scratch_1/epoch_loss_{job_id}.csv"
        out_f3 = f"epoch_loss_scratch_1/validation_loss_{job_id}.csv"

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
                for batch in tqdm(loader) :
                    batch_count +=1
                    if batch_count > limit :
                        break
                    preds = model(batch["x"].to(device),batch["mask"].to(device))
                    labels = batch["labels"].squeeze(0).squeeze(-1)
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
                if scratch_version == 0:
                    out_f4 = f"trained_model_scratch_0/trained_model_{job_id}_{epoch}"
                elif scratch_version == 1:
                    out_f4 = f"trained_model_scratch_1/trained_model_{job_id}_{epoch}"
                torch.save(model.state_dict(),outputs_path + out_f4)
                if early_stopping.stop_training:
                    print(f"Early stopping at epoch {epoch + 1}")
                    print(f"Best validation loss : {early_stopping.best_loss}")
                    break
    ####################################################################################################################

if __name__ == "__main__":
    main()