import uproot
import numpy as np


# load ROOT file and return a dict of numpy arrays using uproot
def load_root_data(file_path, tree_name,sample_size, branches=None):
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        entry_start_=0
        if sample_size == -1 :
            entry_stop_ = tree.num_entries
        else:
            entry_stop_ = sample_size   
        df = tree.arrays(branches,entry_start=entry_start_, entry_stop=entry_stop_, library="np")
      
    return df

# same as in root2hdf5.py
def dict2array_scalar(df_dict) :
    df_array_array = np.array(list(df_dict.values())).T.astype(np.float32)
    return df_array_array

# same as in root2hdf5.py
def dict2array_vector(df_dict) :
    df_list_list_array = np.column_stack(list(df_dict.values()))
    num_jets = len(df_list_list_array)
    num_param = len(df_list_list_array[0])
    df_list_array_array = []
    for i in range(num_jets):
        temp = np.stack([df_list_list_array[i][j] for j in range(num_param)], axis=1).astype(np.float32)
        df_list_array_array.append(temp.T)
    return df_list_array_array

# same as in root2hdf5.py
def trackPt_cut(df_vector,cut_value) :
    jet_number = len(df_vector)
    num_param = df_vector[0].shape[0]
    cut_df_vector = []
    empty_block =np.array([[0] for _ in range(num_param)]) # in case all tracks are removed or no tracks at all
    for i in range(jet_number) :
        num_tracks = df_vector[i].shape[1]
        pt_vec = df_vector[i][0]
        if num_tracks != 0 and np.max(pt_vec) > cut_value :
            if num_tracks>1:
                temp_block = []
                counter = 0
                stop_condition = False
                while np.max(pt_vec) > cut_value:
                    max_idx = np.argmax(pt_vec)
                    pt_vec = np.concatenate([pt_vec[:max_idx],pt_vec[max_idx+1:]])
                    for j in range(num_param):
                        if counter == 0:
                            temp_block.append(np.concatenate([df_vector[i][j][:max_idx],df_vector[i][j][max_idx+1:]]))
                        else:   
                            temp_block.append(np.concatenate([temp_block[num_param*(counter-1) + j][:max_idx],temp_block[num_param*(counter-1) + j][max_idx+1:]]))
                    counter +=1
                    num_tracks = num_tracks - 1  
                    if num_tracks == 1 and np.max(pt_vec) > cut_value :
                        stop_condition = True
                        break
                if stop_condition:
                    cut_df_vector.append(empty_block)          
                else :
                    cut_df_vector.append(np.stack(temp_block[-num_param:],0))
            else:
                cut_df_vector.append(empty_block)
        elif num_tracks != 0:
            cut_df_vector.append(df_vector[i])
        else:
            cut_df_vector.append(empty_block)

    return cut_df_vector

# compute mean and std for scalar features
def get_mean_std_count_scalar(df):
    means = np.mean(df, axis=0)
    stds = np.std(df, axis=0)
    return means, stds

# compute mean and std for vector features, also return mean and std of number of tracks distribution
def get_mean_std_count_vector(df):
    num_features = df[0].shape[0]
    total_sum = np.zeros(num_features)
    total_sqsum = np.zeros(num_features)
    total_count = np.zeros(num_features)
    num_tracks_distib = []

    for matrix in df:
        num_tracks = matrix.shape[1]
        if num_tracks == 1 and matrix[0, 0] == 0:
            continue
        total_sum += np.sum(matrix, axis=1)
        total_sqsum += np.sum(matrix**2, axis=1)
        total_count += matrix.shape[1]
        num_tracks_distib.append(matrix.shape[1])

    means = total_sum / total_count
    stds = np.sqrt(total_sqsum / total_count - means**2)
    nt_mean = np.mean(num_tracks_distib)
    nt_std = np.std(num_tracks_distib)

    return means, stds, nt_mean, nt_std


scalar_branches = [
    "JetEta",
    "JetPhi",
    "JetPt",
    "JetE"
]

vector_branches = [
    "trackPt",
    "trackEta",
    "trackPhi",
    "trackD0",
    "trackZ0",
    "trackTheta",
    "trackDzJet"
]

path = '/home/morandsam/Desktop/internship_25/MAE/data/'
file = "user.cmorenom.45995229._000001.ANALYSIS.root"
tree_name = 'analysis'
out_name = "normalization_first_1e6_000001.ANALYSIS.csv"

# number of events to use for normalization
normalization_sample_size = 1000000

# Pt cut to aply to tracks in MeV
Pt_cut = 25000

# Get mean,std from a sample of size normalization_sample_size with Pt cut applied to tracks
normalization_df_scalar = dict2array_scalar(load_root_data(path + file,'analysis',normalization_sample_size,scalar_branches))
normalization_df_vector = trackPt_cut(dict2array_vector(load_root_data(path + file,'analysis',normalization_sample_size,vector_branches)),Pt_cut)
scalar_means,scalar_stds = get_mean_std_count_scalar(normalization_df_scalar)
vector_means,vector_stds,nt_mean,nt_std = get_mean_std_count_vector(normalization_df_vector)

#output
with open(f"/home/morandsam/Desktop/internship_25/MAE/data/normalization/{out_name}", "w") as f:
    f.write(f"{scalar_means}\n")
    f.write(f"{scalar_stds}\n")
    f.write(f"{vector_means}\n")
    f.write(f"{vector_stds}\n")
    f.write(f"{nt_mean}\n")
    f.write(f"{nt_std}\n")