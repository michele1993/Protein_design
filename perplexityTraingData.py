import os
import torch
import pandas as pd
from dpo_utils import create_preference_pairs, format_for_dpo_trainer
from transformers import  GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, AutoModelForCausalLM, TrainingArguments
from utils import protData_cleaning, insert_char
from datasets import Dataset
import evaluate
import json

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
clean_dataset = protData_cleaning(dataset=dataset, remove_activity_NaN=False)

##Convert data to FASTA file format and add special token 
# 1st  we have to introduce new line characters every 60 amino acids,
# following the FASTA file format.
clean_dataset['mutated_sequence'] = [insert_char(clean_dataset['mutated_sequence'].iloc[i], char='\n', every=60) for i in range(len(clean_dataset))]

# 2nd need to add "<|endoftext|>" token at the beginning and end of each seq
special_token = "<|endoftext|>"
clean_dataset['mutated_sequence'] = [f'{special_token}{s}{special_token}' for s in clean_dataset['mutated_sequence']]

# Get path to dpo fine-tuned and base model
root_dir = os.path.dirname(os.path.abspath(__file__))
dpo_path = os.path.join(root_dir,'dpo_output')
base_path = 'nferruz/ProtGPT2'

# Initialise single tokenizer (i.e., same across models)
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
# Load base vs dpo fine-tuned model

# Compute perplexity for training data across two model
seq = clean_dataset['mutated_seqquence']
perplexity = evaluate.load("perplexity", module_type="metric")

base_perplexity = perplexity.compute(model_id=base_path,
                             predictions=seq,
                             add_start_token=False,
                             batch_size=batch_s,   
                             device=dev)
dpo_perplexity = perplexity.compute(model_id=dpo_path,
                             predictions=seq,
                             add_start_token=False,
                             batch_size=batch_s,   
                             device=dev)

result = {"base_perplexity": base_perplexity['mean_perplexity'],"dpo_perplexity": dpo_perplexity['mean_perplexity']}
# Save to a JSON file
result_dir = os.path.join(root_dir,'results')
os.makedirs(result_dir, exist_ok=True)
with open("TrainingDataPerplexities.json", "w") as file:
    json.dump(result, result_dir)
