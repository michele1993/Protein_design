import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import sys
sys.setrecursionlimit(1500)  # Increase if needed for deeper structures

class ProteinDataset(Dataset):
    """ Dataset class to be passed to DataLoader for protein dataset"""

    def __init__(self, clean_dataset):
        """
        Args:
            csv_file (string): Path to the csv file 
        """
        super().__init__()

        
        # Split amino acids sequence from activations and convert to numpy
        self.amino = clean_dataset['mutated_sequence'].to_numpy()
        self.activation = clean_dataset['activity_dp7'].to_numpy()

    def __len__(self):
        """ Override len method as required"""
        return len(self.amino)

    def __getitem__(self, idx):
        """ Override getitem method as required"""
        # if indexes are tensor, convert to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        amino_item = self.amino[idx]
        activation_item = self.activation[idx]
        
        return amino_item, activation_item
