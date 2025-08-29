import uproot
import awkward as ak
import numpy as np

# Script to interleave two halves of a ROOT file
# Used after prepare_root_files.C to shuffle low and high JVT events


# Input and output files
input_file = "/eos/user/s/smorand/root_cut/data_bal_2.root"
output_file = "/eos/user/s/smorand/root_cut/shuff/data_bal_shuff_2.root"
tree_name = "analysis"  # Replace with your tree name


# Open the ROOT file and read the tree
with uproot.open(input_file) as file:
    tree = file[tree_name]
    data = tree.arrays(library="ak")  # Awkward record array with all branches

# Number of events
n_events = len(data[data.fields[0]])
half = n_events // 2

first_half = np.arange(0, half)
second_half = np.arange(half, n_events)

# Build interleaved indices safely
interleaved = []
for a, b in zip(first_half, second_half):
    interleaved.append(a)
    interleaved.append(b)

# If odd number of events, add the leftover from first_half
if len(first_half) > len(second_half):
    interleaved.append(first_half[-1])

interleaved = np.array(interleaved, dtype=int)

# Apply to all branches
interleaved_dict = {branch: data[branch][interleaved] for branch in data.fields}

# Save to new ROOT file
with uproot.recreate(output_file) as f:
    f[tree_name] = interleaved_dict

print(f"Interleaved ROOT file written to {output_file}")