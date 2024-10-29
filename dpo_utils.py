import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict
import random

def create_preference_pairs(dataset: pd.DataFrame, min_activity_diff: float, prompt:str) -> dict: # pd.DataFrame:
    """     
    Create preference pairs from protein sequences based on their activities.
    
    Args:
        dataset: DataFrame of sequences with corresponding activities.
        min_activity_diff: Minimum difference in activity to consider one sequence preferred
    """
    # Shuffle data just in case
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    dict_dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": []
        }
    for (_, row1), (_, row2) in combinations(dataset.iterrows(),2):
        if abs(row1['activity_dp7'] - row2['activity_dp7']) >= min_activity_diff: # check meet min activity diff
            # Pick sequence with higher activity
            chosen = row1 if row1['activity_dp7'] > row2['activity_dp7'] else row2
            rejected = row2 if row1['activity_dp7'] > row2['activity_dp7'] else row1

            dict_dataset['prompt'].append(prompt)
            dict_dataset['chosen'].append(chosen['mutated_sequence'])
            dict_dataset['rejected'].append(rejected['mutated_sequence'])
    
    return dict_dataset
