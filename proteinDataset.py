import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

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
        self.clean_data = self._basicCleaning(dataset=dataset)

        # Split amino acids sequence from activations and convert to numpy
        self.amino = self.clean_data['mutated_sequence'].to_numpy()
        self.activation = self.clean_data['activity_dp7'].to_numpy()

    def _basicCleaning(self, dataset):
        """ Perform basic data cleaning: remove NaN and duplicates"""

        # identify any row with NaN
        data_nan = pd.isna(dataset).sum(axis=1).astype('bool') # sum to identify if there is at least one NaN entry in a row.
        # remove entries with NaN entirely 
        dataset = dataset.loc[~data_nan, :]

        assert ~pd.isna(dataset).any().any(), "There are NaN entries in the data, need cleaning"

        # Check there are not duplicates in the protein seq which reuqire attention
        duplicats = dataset.duplicated(subset="mutated_sequence")
        assert ~duplicats.any().any(), "There are duplicated protein sequences, investigate and decide how to deal with them"

        # 1st option: remove duplicate based on first occurence, may not be best option depending on type of duplicates 
        #dataset = dataset.drop_duplicates(subset="mutated_sequence", keep="first") 

        return dataset 

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
