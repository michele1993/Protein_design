import os
import torch
import pandas as pd
from dpo_utils import create_preference_pairs
from utils import protData_cleaning, insert_char
from datasets import Dataset
import evaluate
import json

""" 
Investigate whether the dpo fine-tune protGPT2 model gives lower perplexity for 
top 10% activity training sequenses compared to protGPT2 base and SFT models.
"""

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
else:
    dev='cpu'

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir,'dataset','sequences.csv')

# Load all data here since lightweight
dataset = pd.read_csv(file_path)
# Clean data
clean_dataset = protData_cleaning(dataset=dataset, remove_activity_NaN=True)

##Convert data to FASTA file format and add special token 
# 1st  we have to introduce new line characters every 60 amino acids,
# following the FASTA file format.
clean_dataset['mutated_sequence'] = [insert_char(clean_dataset['mutated_sequence'].iloc[i], char='\n', every=60) for i in range(len(clean_dataset))]

# 2nd need to add "<|endoftext|>" token at the beginning and end of each seq
special_token = "<|endoftext|>"
clean_dataset['mutated_sequence'] = [f'{special_token}{s}{special_token}' for s in clean_dataset['mutated_sequence']]

# Get path to dpo fine-tuned and base model
root_dir = os.path.dirname(os.path.abspath(__file__))
base_path = 'nferruz/ProtGPT2' # base model
sft_path = os.path.join(root_dir,'output') #supervised fine-tunes model
dpo_path = os.path.join(root_dir,'dpo_output') #dpo fine-tuned model

# Compute perplexity for top % activity sequences  across two model
top_percentage = 0.05
data = clean_dataset.sort_values('activity_dp7', ascending=False)
top_percent = int(len(data) * top_percentage)
seq = data.head(top_percent)['mutated_sequence']

perplexity = evaluate.load("perplexity", module_type="metric")

base_perplexity = perplexity.compute(model_id=base_path,
                             predictions=seq,
                             add_start_token=False,
                             device=dev)
sft_perplexity = perplexity.compute(model_id=sft_path,
                             predictions=seq,
                             add_start_token=False,
                             device=dev)
dpo_perplexity = perplexity.compute(model_id=dpo_path,
                             predictions=seq,
                             add_start_token=False,
                             device=dev)


# Save as dict to a JSON file
result = {"base_perplexity": base_perplexity['mean_perplexity'], "sft_perplexity": sft_perplexity['mean_perplexity'], "dpo_perplexity": dpo_perplexity['mean_perplexity']}
result_dir = os.path.join(root_dir,'results')
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir,'TrainingDataPerplexities.json')
json.dump(result,open(result_file, 'w'))
