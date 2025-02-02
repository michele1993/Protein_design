import os
import torch
import pandas as pd
from dpo_utils import create_preference_pairs
from transformers import  GPT2Tokenizer, AutoModelForCausalLM
from utils import protData_cleaning, find_longest_common_prefix, insert_char
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import copy
from peft import LoraConfig
import argparse
import logging
#logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs','-e',type=int,nargs='?',default=3) # model type: base, SFT, DPO
# Extract arguments
args = parser.parse_args()
n_epochs = args.epochs

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

# Load SFT model

# Get path to fine-tuned model
root_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root_dir,'output')

tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
model_head = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)#, torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token

# Find longest prefix shared by all sequences for prompt
prompt = find_longest_common_prefix(sequence=list(clean_dataset['mutated_sequence']))
prompt_len = len(tokenizer.encode(prompt))

# Prepare data for DPO
dpo_data_dict = create_preference_pairs(dataset=clean_dataset, min_activity_diff=0.25, prompt=prompt) 
dpo_data = Dataset.from_dict(dpo_data_dict)
LoRA = False


if LoRA:
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
    )
else:
    peft_config=None

training_args = DPOConfig(
    overwrite_output_dir=True,
    output_dir="dpo_output", 
    num_train_epochs= n_epochs,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    learning_rate=1e-6,
    push_to_hub=False,
    max_length=150+prompt_len,# based on training data
    max_prompt_length=prompt_len,
)
trainer = DPOTrainer(
    model=model_head,
    ref_model=None,
    peft_config=peft_config,  
    args=training_args,
    train_dataset=dpo_data,
    tokenizer=tokenizer,
    beta=0.1
)

trainer.train()
# save model at the end of training
trainer.save_model()
