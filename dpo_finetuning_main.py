import os
import torch
import pandas as pd
from dpo_utils import create_preference_pairs, format_for_dpo_trainer
from transformers import  GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, AutoModelForCausalLM
from utils import protData_cleaning, find_longest_common_prefix, insert_char
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import copy

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

##Convert data to FASTA file format and add special token 
# 1st  we have to introduce new line characters every 60 amino acids,
# following the FASTA file format.
clean_dataset['mutated_sequence'] = [insert_char(clean_dataset['mutated_sequence'].iloc[i], char='\n', every=60) for i in range(len(clean_dataset))]


# 2nd need to add "<|endoftext|>" token at the beginning and end of each seq
special_token = "<|endoftext|>"
clean_dataset['mutated_sequence'] = [f'{special_token}{s}{special_token}' for s in clean_dataset['mutated_sequence']]

# Find longest prefix shared by all sequences for prompt
prompt = find_longest_common_prefix(sequence=list(clean_dataset['mutated_sequence']))

# Prepare data for DPO
#preferences = create_preference_pairs(dataset=clean_dataset, min_activity_diff=0.1) 
#dpo_data = format_for_dpo_trainer(pairs_df=preferences, prompt=prompt)
dpo_data_dict = create_preference_pairs(dataset=clean_dataset, min_activity_diff=0.1, prompt=prompt) 
dpo_data = Dataset.from_dict(dpo_data_dict)

# Load SFT model
# Get path to fine-tuned model
root_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root_dir,'output')

#model_path = "nferruz/ProtGPT2"
model_path = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
#model_head = GPT2LMHeadModel.from_pretrained(model_path)
model_head = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

training_args = DPOConfig(output_dir="dpo_output", 
                          logging_steps=10,
                          per_device_train_batch_size=1,
                          gradient_accumulation_steps=1,
                          gradient_checkpointing=True,
                          learning_rate=5e-5,
                          remove_unused_columns=False,
                          max_length=450,
                          bf16=True
                         )
trainer = DPOTrainer(
    model=model_head,
    ref_model=copy.deepcopy(model_head),
    args=training_args,
    train_dataset=dpo_data,
    tokenizer=tokenizer,
    beta=0.1,
)

trainer.train()
