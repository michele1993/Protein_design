import os
import torch
import pandas as pd
from dpo_utils import create_preference_pairs, format_for_dpo_trainer
from utils import protData_cleaning, find_longest_common_prefix
from trl import DPOTrainer

# Useful variables
batch_size = 10

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir,'dataset','sequences.csv')

# Load all data here since lightweight
dataset = pd.read_csv(file_path)
# Clean data
clean_dataset = protData_cleaning(dataset=dataset, remove_activity_NaN=True)

#trial = ['ABCDEFGHILDDHSBEJDNNSNDJF','ABCDEFGHILDdhsbdsh', 'ABCDEFGHILDndnsndnsj', 'ABCDEFGHILDndjsndj']

#TODO: DATA need to be in FASTA format!

# Find longest prefix shared by all sequences for prompt
prompt = find_longest_common_prefix(sequence=list(clean_dataset['mutated_sequence']))

# Prepare data for DPO
preferences = create_preference_pairs(dataset=clean_dataset, min_activity_diff=0.1) 
dpo_data = format_for_dpo_trainer(pairs_df=preferences, prompt=prompt)

#trainer = DPOTrainer(
#    model,
#    ref_model,
#    beta=0.1,
#    train_dataset=dpo_samples,
#    # ... other DPOTrainer parameters
#)
