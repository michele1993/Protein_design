import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict
import random

def create_preference_pairs(dataset: pd.DataFrame, min_activity_diff: float = 0.1) -> pd.DataFrame:
    """     
    Create preference pairs from protein sequences based on their activities.
    
    Args:
        dataset: DataFrame of sequences with corresponding activities.
        min_activity_diff: Minimum difference in activity to consider one sequence preferred
    """
    # Shuffle data just in case
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    pairs = []
    for (idx1, row1), (idx2, row2) in combinations(dataset.iterrows(),2):
        if abs(row1['activity_dp7'] - row2['activity_dp7']) >= min_activity_diff: # check meet min activity diff
            # Pick sequence with higher activity
            chosen = row1 if row1['activity_dp7'] > row2['activity_dp7'] else row2
            rejected = row2 if row1['activity_dp7'] > row2['activity_dp7'] else row1

            pairs.append({
                'chosen': chosen['mutated_sequence'],
                'rejected': rejected['mutated_sequence'],
                'chosen_activity': chosen['activity_dp7'],
                'rejected_activity': rejected['activity_dp7']
            })
    
    return pd.DataFrame(pairs)

def format_for_dpo_trainer(pairs_df: pd.DataFrame, prompt: str) -> List[Dict]:
    """
    Format preference pairs for TRL DPOTrainer.

    Args:
        pairs_df: DataFrame with chosen and rejected sequences
        prompt: Prompt string to prepend to sequences
    """
    #dpo_samples = []
    #for _, row in pairs_df.iterrows():
    #    sample = {
    #        "prompt": prompt,
    #        "chosen": row['chosen'],
    #        "rejected": row['rejected']
    #    }
    #    dpo_samples.append(sample)
    dpo_dict = {
            "prompt": prompt,
            "chosen": row['chosen'],
            "rejected": row['rejected']
    }


    return dpo_samples
