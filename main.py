from proteinDataset import ProteinDataset
from torch.utils.data import DataLoader
from tokenization import Seq_Label_Tokenization
from utils import protData_cleaning
import os
import torch
import pandas as pd

# Useful variables
batch_size = 10

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir,'dataset','sequences.csv')

# Load all data here since lightweight
dataset = pd.read_csv(file_path)

## Apply basic data cleaning
clean_data = protData_cleaning(dataset=dataset, remove_activity_NaN=True)

## Divide training and validation data
shuffled_data = clean_data.sample(frac=1).reset_index(drop=True) # shuffle data

# Select % of validation seqs (i.e., 90/10)
n_seq = len(shuffled_data)
n_validation = n_seq // 10

# Select training and validation seqs
validation_data = shuffled_data.iloc[:n_validation,:].copy()
training_data = shuffled_data.iloc[n_validation:,:].copy()

# Prepare data for training model
training_data = ProteinDataset(clean_dataset=training_data) # Dataset class for DataLoader
validation_data = ProteinDataset(clean_dataset=validation_data)

# Initalise tokenizer
tokenizer = Seq_Label_Tokenization(clean_dataset=training_data, aminoacid_data=True)

# Intialise dataloader with appropriate collate_fn from tokenizer class
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=tokenizer.collate_fn)

for seq, activity in dataloader:
    print(seq.shape)
    print(activity.shape)
