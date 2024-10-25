import os
import pandas as pd
import numpy as np
import torch
from utils import protData_cleaning, insert_char

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir,'dataset','sequences.csv')

dataset = pd.read_csv(file_path)

## ------ 1st Clean data -----
data_cleaned = protData_cleaning(dataset=dataset, remove_activity_NaN=False)

## ------- 2nd Divide data in training and validation and add tokens ----
# for the moment ingore the activations just with all sequences
data_cleaned_seq = data_cleaned.iloc[:,0]

# 3rd need to slip the data in training and validation
# Select % of validation seqs (i.e., 90/10)
n_seq = len(data_cleaned_seq)
n_validation = n_seq // 10

# Select n. random indexes for validation 
val_indx = np.random.randint(0, n_seq, n_validation)
# Select validation seqs
val_seq = data_cleaned_seq.iloc[val_indx].copy()
# Select training seqs by eliminating validation seqs
training_seq = data_cleaned_seq.drop(index=val_indx)

## --------- 3rd Convert data to FASTA file format and add special token ------------
# 1st  we have to introduce new line characters every 60 amino acids,
# following the FASTA file format.
training_seq = [insert_char(training_seq.iloc[i], char='\n', every=60) for i in range(len(training_seq))]
val_seq = [insert_char(val_seq.iloc[i], char='\n', every=60) for i in range(len(val_seq))]

# 2nd need to add "<|endoftext|>" token at the beginning and end of each seq
special_token = "<|endoftext|>"
training_seq = [ f'{special_token}{s}{special_token}' for s in training_seq]
val_seq = [ f'{special_token}{s}{special_token}' for s in val_seq]

# 3rd: concatane all strings together and save in a txt file
training_concatenated = ''.join(training_seq)
val_concatenated = ''.join(val_seq)

## ----- 4th Save concatenated strings -----
with open('training.txt','w') as file:
    file.write(training_concatenated)

with open('validation.txt','w') as file:
    file.write(val_concatenated)
