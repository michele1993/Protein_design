import os
import pandas as pd
import numpy as np
import torch

# Define helper method
def insert_char(seq, every=60):
    """
    Insert a character into a string every `n` characters.
    """
    return '\n'.join(seq[i:i+every] for i in range(0, len(seq), every))

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir,'dataset','sequences.csv')

dataset = pd.read_csv(file_path)


## ------ 1st Remove any pair if contains NaN -----
# identify any row with NaN
data_nan = pd.isna(dataset).sum(axis=1).astype('bool') # sum to identify if there is at least one NaN entry in a row.
# remove entries with NaN entirely 
data_cleaned = dataset.loc[~data_nan, :]

assert ~pd.isna(data_cleaned).any().any(), "There are NaN entries in the data, need cleaning"
## ----------------------------

## ------ 2nd Remove any duplicate entry ----
# Check there are not duplicates in the protein seq which reuqire attention
duplicats = data_cleaned.duplicated(subset="mutated_sequence")
assert ~duplicats.any().any(), "There are duplicated protein sequences, investigate and decide how to deal with them"

# 1st option: remove duplicate based on first occurence, may not be best option depending on type of duplicates 
#data_cleaned = data_cleaned.drop_duplicates(subset="mutated_sequence", keep="first") 
## -------------------------------------------

## ------- 3rd Divide data in training and validation and add tokens ----
# for the moment ingore the activations just with all sequences
data_cleaned_seq = data_cleaned.iloc[:,0]

# 3rd need to slip the data in training and validation
# Select % of validation seqs (i.e., 90/10)
n_seq = len(data_cleaned_seq)
n_validation = n_seq // 10

# Select n. random indexes for validation 
val_indx = np.random.randint(0, n_seq, n_validation)
# Select validation seqs
val_seq = data_cleaned_seq.iloc[val_indx]
# Select training seqs by eliminating validation seqs
training_seq = data_cleaned_seq.drop(index=val_indx)

# 1st  we have to introduce new line characters every 60 amino acids,
# following the FASTA file format.
training_seq = [insert_char(training_seq.iloc[i]) for i in range(len(training_seq))]
val_seq = [insert_char(val_seq.iloc[i]) for i in range(len(val_seq))]

# 2nd need to add "<|endoftext|>" token at the beginning and end of each seq
special_token = "<|endoftext|>"
training_seq = [ f'{special_token}{s}{special_token}' for s in training_seq]
val_seq = [ f'{special_token}{s}{special_token}' for s in val_seq]


# 3rd: concatane all strings together and save in a txt file
training_concatenated = ''.join(training_seq)
val_concatenated = ''.join(val_seq)

# Save concatenated strings
with open('training.txt','w') as file:
    file.write(training_concatenated)

with open('validation.txt','w') as file:
    file.write(val_concatenated)
