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
from accelerate import Accelerator
from datetime import datetime
import pytz

##################################################################################################################
## Some general functions ########################################################################################
##################################################################################################################

# randomly mask a fraction of real tracks in the jet
def random_masking(
    mask: np.ndarray,
    mask_fraction: float = 0.5,
    seed: int = None,
) -> np.ndarray:
    """Randomly drop a fraction of the jet based on the total number of constituents."""
    # Create the random number generator, the number to drop and the mask
    rng = np.random.default_rng(seed)
    max_drop = np.ceil(mask_fraction * mask.sum()).astype(int) 
    null_mask = np.full_like(mask, False)
    # Exit now if we are not dropping any nodes
    if max_drop == 0:
        return null_mask
    # Generate a random number per node, the lowest frac will be killed
    rand = rng.uniform(size=len(mask))
    rand[~mask] = 9999  # Padded nodes shouldnt be dropped
    # Get the indices for which to sort the random values
    drop_idx = np.argsort(rand)[:max_drop]
    # Create the null mask by dropping the nodes
    null_mask[drop_idx] = True
    return null_mask

def main():

    # create dataloader from h5 files in a folder, batch are created directly in H5Dataset class
    def create_dataloader(folder_path ,batch_size, shuffle=False):
        dataset = H5Dataset(folder_path,batch_size)
        # num_workers is the number of CPU subprocesses to use for data loading
        # prefetch_factor is the number of samples loaded in advance by each worker
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers = int(sys.argv[3]), prefetch_factor=8)

    ##################################################################################################################
    ## Class declarations ############################################################################################
    ##################################################################################################################

    # create transformer encoder with a given number of layers and heads
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
            
            # ignore padding tokens, ! logic is reversed because True means real track and False means padding in the rest of the code
            src_key_padding_mask = ~mask.bool()  # True = ignore
            return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    # look at slides for explanation of the architecture
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
            replace_mask = (~null_mask & mask)  # [batch, feature_number]
            extra_subset = self.extra_token.expand(latent_space.shape[0], -1, -1)
            latent_space = torch.where(replace_mask.unsqueeze(-1), extra_subset, latent_space)
            decoded = self.decoder(latent_space,mask)
            out = self.unembedding(decoded)  # Unembedding the latent space back to the original feature space
            return out

    # dataloader class to load batches directly from h5 files in a folder
    # each file is opened only once and closed when another file is needed
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

            # mask corresponds to real tracks in the jet (1) and padding (0)
            masks = self.file["mask"][start:end]==1 # convert to boolean
            
            # null_mask corresponds to the randomly masked tracks in the jet and padding
            null_mask = np.stack([random_masking((i==1), 0.5) for i in masks])

            output ={"x": torch.tensor(batch).float(), "null_mask": torch.tensor(null_mask).bool(), 
            "mask": torch.tensor(masks).bool()} 

            return output
        
    ###################################################################################################################
    ## Inputs and data ################################################################################################
    ###################################################################################################################

    folder_path = "/eos/user/s/smorand/h5data/v3/"
    folder_path_valid = "/eos/user/s/smorand/h5data/v3_validation/"
    outputs_path = "/eos/user/s/smorand/outputs/"
    batch_size = 1000

    # job_id for outputs files
    job_id = sys.argv[1]

    # number of layers in Transformer
    nlayers = int(sys.argv[2])

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
    model = TMAE(features_number=12, latent_space_dim=128, nlayer=nlayers, nhead=4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(f"Ready for training.")
    
    n_parameters = 0
    for i in model.parameters():
        n_parameters += i.numel() if i.requires_grad else 0
    print(f"Trainable parameters : {n_parameters}.")

    with open(outputs_path+f"batch_loss/batch_loss_T_{job_id}.csv", "w") as f1, open(outputs_path+f"epoch_loss/epoch_loss_T_{job_id}.csv", "w") as f2, open(outputs_path + f"epoch_loss/validation_loss_T_{job_id}.csv", "a") as f3:
        for epoch in range(6):
            model.train()
            total_loss = 0
            batch_count = 0
            for batch in loader:
                batch_count +=1
                preds = model(**batch)
                mask = batch["mask"][0].to(device)     
                null_mask = batch["null_mask"][0].to(device) 
                x = batch["x"].to(device)
                loss_mask = mask & ~null_mask
                loss = loss_fn(preds[loss_mask], x[0][loss_mask])
                optimizer.zero_grad() # reset gradients
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                f1.write(f"{loss.item()}\n")
                # print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
                # print(f"Processed batch {batch_count} of epoch {epoch + 1}")
            avg_loss = total_loss/batch_count
            print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")
            f2.write(f"{avg_loss}\n")

            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in loader_validation:
                    val_batches += 1
                    preds = model(**batch)
                    mask = batch["mask"][0].to(device)     
                    null_mask = batch["null_mask"][0].to(device) 
                    x = batch["x"].to(device)
                    loss_mask = mask & ~null_mask
                    loss = loss_fn(preds[loss_mask], x[0][loss_mask])
                    val_loss += loss.item()
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('nan')
            print(f"Epoch {epoch + 1} | Validation Loss: {avg_val_loss:.4f}")
            f3.write(f"{avg_val_loss}\n")

            torch.save(model.state_dict(),outputs_path + f"trained_model/trained_model_T_{job_id}_{epoch}.pth")
    ####################################################################################################################

if __name__ == "__main__":
    main()