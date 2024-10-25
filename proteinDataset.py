import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from utils import protData_cleaning
import sys
sys.setrecursionlimit(1500)  # Increase if needed for deeper structures

class ProteinDataset(Dataset):
    """ Dataset class to be passed to DataLoader for protein dataset"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file 
        """
        super().__init__()

        # Load all data here since lightweight
        dataset = pd.read_csv(csv_file)

        ## Apply basic data cleaning
        self.clean_data = protData_cleaning(dataset=dataset, remove_activity_NaN=True)

        # Split amino acids sequence from activations and convert to numpy
        self.amino = self.clean_data['mutated_sequence'].to_numpy()
        self.activation = self.clean_data['activity_dp7'].to_numpy()

    def __len__(self):
        """ Override len method as required"""
        return len(self.clean_data)

    def __getitem__(self, idx):
        """ Override getitem method as required"""
        # if indexes are tensor, convert to list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        amino_item = self.amino[idx]
        activation_item = self.activation[idx]
        
        return amino_item, activation_item
