from proteinDataset import ProteinDataset
from torch.utils.data import DataLoader
from tokenization import Seq_Label_Tokenization
import os
import torch

# Useful variables
batch_size = 10

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir,'dataset','sequences.csv')
clean_data = ProteinDataset(csv_file=file_path) # taylord Dataset class, which already cleans the data 

# Initalise tokenizer
tokenizer = Seq_Label_Tokenization(clean_dataset=clean_data, aminoacid_data=True)

# Intialise dataloader with appropriate collate_fn from tokenizer class
dataloader = DataLoader(clean_data, batch_size=batch_size, shuffle=True, collate_fn=tokenizer.collate_fn)

for seq, activity in dataloader:
    print(seq.shape)
    print(activity.shape)
