import sys
import os
import torch
from tqdm import tqdm
import h5py
import torch.nn as nn
import numpy as np
from torchvision.ops import MLP
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pytz

##################################################################################################################
## Some general functions ########################################################################################
##################################################################################################################

# randomly mask a fraction of real tracks in the jet
def random_masking(
    mask: np.ndarray,
    mask_fraction: float = 0.4,
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

    # look at slides for explanation of the architecture
    class DeepSets(nn.Module):
        def __init__(self, inputs_number, inter12_number, hidden_layers ,outputs_number, loop_number): # doesn not depend on tokens number
            super().__init__()

            self.loop_number = loop_number
            self.inputs_number = inputs_number
            self.outputs_number = outputs_number

            hidden_layer_mlp1 = []
            hidden_layer_mlp2 = []
            hidden_layers_final = []

            for i in range(len(hidden_layers)):
                hidden_layer_mlp1.append(hidden_layers[i])
                hidden_layer_mlp2.append(hidden_layers[i])
                hidden_layers_final.append(hidden_layers[i])
            hidden_layer_mlp1.append(inter12_number)
            hidden_layers_final.append(outputs_number)

            self.mlp_blocks_1 = nn.ModuleList()
            self.mlp_blocks_2 = nn.ModuleList()

            for i in range(loop_number):
                self.mlp_blocks_1.append(MLP(in_channels=int(2**i)*inputs_number, hidden_channels=hidden_layer_mlp1, activation_layer=nn.ReLU).to(device))
                self.mlp_blocks_2.append(MLP(in_channels=inter12_number, hidden_channels=sum([hidden_layer_mlp2,[int(2**i)*inputs_number]],[]), activation_layer=nn.ReLU).to(device))

            self.final_mlp = MLP(in_channels=int(2**(loop_number))*inputs_number, hidden_channels=hidden_layers_final, activation_layer=nn.ReLU).to(device)

        def one_loop_forward(self, x, mask_unsqueezed, index):
            mlp_1 = self.mlp_blocks_1[index]
            x_mlp1 = mlp_1(x) 
            x_mlp1 = x_mlp1 * mask_unsqueezed        # apply mask before pooling
            pooled = x_mlp1.mean(dim=1)              # average pooling on token number
            mlp_2 = self.mlp_blocks_2[index]
            x_mlp2 = mlp_2(pooled)
            x_mlp2 = x_mlp2.unsqueeze(1)
            x_skip = torch.cat([x_mlp2.repeat(1,x.size(1),1), x], dim=2)  # concatenate pooled and mlp2 output
            return x_skip

        def forward(self, x, mask):               
            mask_unsqueezed = mask.unsqueeze(-1)     
            for i in range(self.loop_number):
                x = self.one_loop_forward(x, mask_unsqueezed,i)
            out = self.final_mlp(x)
            return out

    # look at slides for explanation of the architecture
    class MAE(nn.Module):
        def __init__(self,features_number, latent_space_dim ,inter12_number_DS1, hidden_layers_DS1 , inter12_number_DS2, hidden_layers_DS2, loop_number_DS1, loop_number_DS2): # doesn not depend on tokens number
            super().__init__()

            self.encoder = DeepSets(inputs_number=features_number, inter12_number=inter12_number_DS1, hidden_layers=hidden_layers_DS1, outputs_number=latent_space_dim, loop_number=loop_number_DS1)
            self.decoder = DeepSets(inputs_number=latent_space_dim, inter12_number=inter12_number_DS2, hidden_layers=hidden_layers_DS2, outputs_number=features_number, loop_number=loop_number_DS2)

            self.extra_token = nn.Parameter(torch.randn(1, 35, latent_space_dim,device=device))

        def forward(self,x, mask, null_mask):
            x = x[0].to(device)
            mask = mask[0].to(device)
            null_mask = null_mask[0].to(device)
            latent_space = self.encoder(x, null_mask)
            replace_mask = (~null_mask & mask)
            extra_subset = self.extra_token.expand(latent_space.shape[0], -1, -1)
            latent_space = torch.where(replace_mask.unsqueeze(-1), extra_subset, latent_space)
            out = self.decoder(latent_space,mask)
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

    # paths training dataset (h5 files)
    folder_path = "/eos/user/s/smorand/h5data/v3/"
    # paths validation dataset (h5 files)
    folder_path_valid = "/eos/user/s/smorand/h5data/v3_validation/"
    outputs_path = "/eos/user/s/smorand/outputs/"

    batch_size = 1000

    # job_id for outputs files
    job_id = sys.argv[1]

    # number of convolution loops in DeepSets
    loop_number_ = int(sys.argv[2])

    ####################################################################################################################
    ## Model and training ##############################################################################################
    ####################################################################################################################
    tz = pytz.timezone("Europe/Zurich")
    
    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    # use GPU if available (for training and validation not for dataloader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print("Creation of Dataloader...")
    loader = create_dataloader(folder_path, batch_size=batch_size, shuffle=False)
    loader_validation = create_dataloader(folder_path_valid, batch_size=batch_size, shuffle=False)

    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print("Initialization of model...")
    model = MAE(12, 128, 128, [256], 128, [256], loop_number_DS1=loop_number_, loop_number_DS2=loop_number_)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    print(datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z%z"))
    print(f"Ready for training.")
    
    n_parameters = 0
    for i in model.parameters():
        n_parameters += i.numel() if i.requires_grad else 0
    print(f"Trainable parameters : {n_parameters}.")

    with open(outputs_path+f"batch_loss/batch_loss_{job_id}.csv", "w") as f1, open(outputs_path+f"epoch_loss/epoch_loss_{job_id}.csv", "w") as f2, open(outputs_path + f"epoch_loss/validation_loss_{job_id}.csv", "a") as f3:
        for epoch in range(5):
            model.train()
            total_loss = 0
            batch_count = 0
            for batch in loader :
                batch_count +=1
                preds = model(**batch)
                mask = batch["mask"][0].to(device)     
                null_mask = batch["null_mask"][0].to(device) 
                x = batch["x"].to(device)
                # create mask that select only masked real tracks (not padding) to compute loss
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
                for batch in loader_validation :
                    preds = model(**batch)
                    mask = batch["mask"][0].to(device)     
                    null_mask = batch["null_mask"][0].to(device) 
                    x = batch["x"].to(device)
                    loss_mask = mask & ~null_mask
                    loss = loss_fn(preds[loss_mask], x[0][loss_mask])
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('nan')
            print(f"Epoch {epoch + 1} | Validation Loss: {avg_val_loss:.4f}")
            f3.write(f"{avg_val_loss}\n")

            torch.save(model.state_dict(),outputs_path + f"trained_model/trained_model_{job_id}_{epoch}.pth")
    ####################################################################################################################

if __name__ == "__main__":
    main()