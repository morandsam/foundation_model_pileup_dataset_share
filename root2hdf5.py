import uproot
import h5py
import sys
import os
import numpy as np
import torch


# convert dict made of scalar features to arrays 
def dict2array_scalar(df_dict) :
    df_array_array = np.array(list(df_dict.values())).T.astype(np.float32)
    return df_array_array

# convert dict made of vector features to arrays
def dict2array_vector(df_dict) :
    df_list_list_array = np.column_stack(list(df_dict.values()))
    num_jets = len(df_list_list_array)
    num_param = len(df_list_list_array[0])
    df_list_array_array = []
    for i in range(num_jets):
        temp = np.stack([df_list_list_array[i][j] for j in range(num_param)], axis=1).astype(np.float32)
        df_list_array_array.append(temp.T)
    return df_list_array_array

# remove tracks with Pt > cut_value, if all tracks are removed or no tracks at all, replace by an empty block
# can be optimized
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

# normalize scalar features given mean and std
def normalize_scalar(df,means,stds):
    df = (df - means[None,:])/stds[None,:]
    return df

# normalize vector features given mean and std
def normalize_vector(df,means,stds):
    means = np.array(means)
    stds = np.array(stds)
    for i, matrix in enumerate(df):
        if matrix.shape[1] == 1 and matrix[0, 0] == 0:
            continue
        df[i] = (matrix - means[:, None]) / stds[:, None]
    return df    

# concatenate scalar and vector features, add number of tracks as extra scalar feature (capped at 35)
# at the end of each track, scalar features are repeated
def concatenate_scalar_vector(df_scalar,df_vector):
    X_total = []
    num_jets = len(df_scalar)
    for i in range(num_jets):
        num_tracks = len(df_vector[i][0])
        vectors = np.stack(df_vector[i], axis=1).astype(np.float32)
        scalars = df_scalar[i][None, :].astype(np.float32)
        if num_tracks > 35 :
            num_tracks_padded = 35
        else :
            num_tracks_padded = num_tracks
        num_tracks_padded = (num_tracks_padded - nt_mean)/nt_std
        num_tracks_vec = np.ones((1,1))*num_tracks_padded
        scalars_ext = np.concatenate([scalars,num_tracks_vec],axis=1)   
        scalars = scalars_ext.repeat(num_tracks, axis=0)    
        full_jet_infos = np.concatenate([vectors, scalars], axis=1)
        full_jet_infos = full_jet_infos[np.argsort(full_jet_infos[:, 0])[::-1]]
        X_total.append(full_jet_infos.copy())

    return X_total

# pad sequences to the same length which can be smaller than the longest sequence
# improvement of pad_sequence from torch.nn.utils.rnn
def pad_to_length(sequences, target_length, padding_value=0.0):
    batch_size = len(sequences)
    feature_dim = sequences[0].size(-1)

    padded = torch.full((batch_size, target_length, feature_dim), padding_value, dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        length = min(seq.size(0), target_length)
        padded[i, :length] = seq[:length]
    return padded

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


# folder containing root files
path = "/eos/user/s/smorand/root_cut/shuff/"
all_files = sorted(f for f in os.listdir(path) if f.endswith('.root'))

# name of the tree inside the root files
tree_name = 'analysis'

# number of jets in each output h5 file
chunk_size = 1000000

# Pt cut value for tracks in MeV
Pt_cut = 25000

# index of the root file and chunk to be processed in the root file folder
file_index = int(sys.argv[1])
chunk_index = int(sys.argv[2])

file = uproot.open(path + all_files[file_index])
tree = file[tree_name]
total_entries = tree.num_entries

# total number of chunks in the current root file
total_chunks = (total_entries + chunk_size - 1) // chunk_size

# process only if the chunk index is valid
if chunk_index < total_chunks and file_index < len(all_files) :

    print(f"Start processing chunk {chunk_index+1}/{total_chunks} of file {file_index}")

    # normalization parameters computed on a separate sample
    scalar_means = np.array([-2.4184727e-03,  8.3196737e-02,  2.3598588e+04,  5.5035141e+04])
    scalar_stds = np.array([1.4241284e+00, 1.7929353e+00, 1.4051542e+04, 4.8652137e+04])
    vector_means = np.array([ 1.61204979e+03, -5.34619595e-04,  8.26471817e-02, -7.84530174e-02, -1.29296318e-01,  1.56633096e+00,  5.89444117e+01])
    vector_stds = np.array([2.26822132e+03, 1.37915302e+00, 1.79648146e+00, 2.22713018e+00, 5.73022644e+01, 1.00064207e+00, 4.92899774e+01])
    nt_mean = 26.740283
    nt_std = 12.115275980344446
    
    entry_start = chunk_index * chunk_size
    entry_stop = min(entry_start + chunk_size, tree.num_entries)
    arrays = tree.arrays(entry_start=entry_start, entry_stop=entry_stop,library="np")
    dict_scalar = {k:arrays[k] for k in scalar_branches}
    dict_vector = {k:arrays[k] for k in vector_branches}
    dict_label = {"JVT":arrays["JetJVT"]}
    scalar_array = normalize_scalar(dict2array_scalar(dict_scalar),scalar_means,scalar_stds)
    vector_array = normalize_vector(trackPt_cut(dict2array_vector(dict_vector),Pt_cut),vector_means,vector_stds)
    labels = dict2array_scalar(dict_label)
    X_total = concatenate_scalar_vector(scalar_array,vector_array)
    padded_tokens = pad_to_length([torch.tensor(i) for i in X_total],35)
    mask = torch.zeros(padded_tokens.shape[:2], dtype=torch.float32) # [batch_size, token_number]
    for i, s in enumerate(X_total):
        mask[i, :s.shape[0]] = 1.0
    # save to h5 file
    with h5py.File(f"/eos/user/s/smorand/h5data/v3_cut/05_05/file_{file_index}_chunk_{chunk_index}.h5", "w") as f:
        f.create_dataset("jets",data=padded_tokens)
        f.create_dataset("mask",data=mask)
        f.create_dataset("labels",data=labels)

