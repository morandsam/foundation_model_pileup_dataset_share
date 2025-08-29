#########################################################################
List of command to execute the different scripts and main purpose of them
#########################################################################

#########

compute_normalization.py : compute std,mean of scalar and vector features from a given rootfile, write their value in an output file

command : python3 compute_normalization.py

#########

functions.py : contain two functions that can be useful to interpret results from classification head, can't be exectued like that

#########

prepare_root_files.C : take a root file  and apply a given JVT cut on it. Create a new rootfile with JVT cut and perfect class balance. data are not shuffle ! need to use suffle_root.py to do that.

command : root prepare_root_files.C

#########

shuffle_root.py : take the output of prepare_root_files.C and interleaved data so that every odd event correspond to low JVT and even ones to high JVT

command : python3 shuffle_root.py

#########

root2hdf5.py : take as input a folder of root files and convert a chunk of one of them to an hdf5 file applying data preparation. data preparation is described in attached slides. 

command : python3 root2hdf5.py file_index chunck_index

file_index : int --> which file in the folder
chunck_index : int --> given file_index, the file is made of n parts of size chunk_size. chunck_index represents     which part of this file is prepared and converted to hdf5 file

#########

MAE.py : pretraining of MAE using DeepSet. take as input a folder of training hdf5 files and validation hdf5 and write/save model/results in output files

command : python3 MAE.py job_id convolution_number n_workers

job_id : name for outputs (useful with HTCondor to distinguish outputs)
convolution_number : number of convolution in the Deepsets. it's the same for encoder and decoder
n_workers : number of CPUs subprocesses to allocate for dataloading (in HTCondor corresponds to number of CPUs requested and locally to the number of CPU cores you want to allocate to the job)

#########

TMAE.py : pretraining of MAE using Transformer. take as input a folder of training hdf5 files and validation hdf5 and write/save model/results in output files

command : python3 TMAE.py job_id n_layers n_workers

job_id : name for outputs (useful with HTCondor to distinguish outputs)
n_layers : number of layers in transformer. same for encoder and decoder
n_workers : number of CPUs subprocesses to allocate for dataloading (in HTCondor corresponds to number of CPUs requested and locally to the number of CPU cores you want to allocate to the job)

#########

JVT_classification.py : training of JVT classification using (un)frozen and (un)pre-trained transformer encoder

command : python3 JVT_classification.py job_id n_workers pretrained batch_limit

job_id : name for outputs (useful with HTCondor to distinguish outputs)
n_workers : number of CPUs subprocesses to allocate for dataloading (in HTCondor corresponds to number of CPUs requested and locally to the number of CPU cores you want to allocate to the job)
pretrained : bool --> use of (un)pretrained Transformer encoder
batch_limit : int --> number of batch of size batch_size considered for training (enable to select only a subset of whole data)

#########

JVT_scratch.py : training of JVT classification using simple MLP
command : python3 JVT_scratch.py job_id n_workers scratch_model batch_limit

job_id : name for outputs (useful with HTCondor to distinguish outputs)
n_workers : number of CPUs subprocesses to allocate for dataloading (in HTCondor corresponds to number of CPUs requested and locally to the number of CPU cores you want to allocate to the job)
scratch_model : bool --> 0 for model 0 and 1 for model 1
batch_limit : int --> number of batch of size batch_size considered for training (enable to select only a subset of whole data)
