import torch
from transformers import pipeline, GPT2Tokenizer, AutoModelForCausalLM 
import os
from torch.distributions import Categorical
import torch.nn.functional as F
import pandas as pd
import evaluate
import argparse

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
else:
    dev='cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--model-name','-m',type=str,nargs='?',default='dpo') # model type: base, SFT, DPO
parser.add_argument('--n-sequences','-s',type=int,nargs='?',default=1000) # n. of sequences generated
# Extract arguments
args = parser.parse_args()
model_type = args.model_name
n_sequences = args.n_sequences

# Select fine-tuned model for inference
root_dir = os.path.dirname(os.path.abspath(__file__))

if model_type == 'sft':
    model_path = os.path.join(root_dir,'output')
elif model_type == 'dpo':
    model_path = os.path.join(root_dir,'dpo_output')
elif model_type == 'base':
    model_path = 'nferruz/ProtGPT2'
else:
    raise ValueError('Need to select a valid model type for inference')

# Initialise fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
#ft_protgpt2_generator = pipeline('text-generation', model=model_path, tokenizer=tokenizer, device=dev)

# Generate sequences
prompt = "<|endoftext|>ATAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPI"
#with torch.no_grad():
#    sequences = ft_protgpt2_generator(prompt, 
#                                  max_length=425, 
#                                  do_sample=True, 
#                                  top_k=950, 
#                                  repetition_penalty=1.2, 
#                                  num_return_sequences=n_sequences, 
#                                  eos_token_id=tokenizer.eos_token_id,
#                                  batch_size=1,
#                                  use_cache=False
#                                 )
# Free-up GPU memory by deleting generator
#del ft_protgpt2_generator
#torch.cuda.empty_cache()

# Convert to list of string
#sequences = [d['generated_text'] for d in sequences]

# Initialise model head 
model = AutoModelForCausalLM.from_pretrained(model_path).to(dev)

# Compute perplexity for each seq and store it with the corresponding seq
perplexity = evaluate.load("perplexity", module_type="metric")
dict_predictions = {
        "mutated_sequence": [],
        "perplexity": [],
        }
batch_s = 30
with torch.no_grad():
    for i in range(0, n_sequences, batch_s):
        # Generate a sequence
        batch_prompt = [prompt] * batch_s
        tokenizer_output = tokenizer(batch_prompt, return_tensors="pt", padding=True)
        inputs = tokenizer_output.input_ids.to(dev)
        attention_mask = tokenizer_output.attention_mask.to(dev)
        seq_indx = model.generate(inputs, 
                                  pad_token_id=tokenizer.eos_token_id, 
                                  eos_token_id= tokenizer.eos_token_id,
                                  max_new_tokens=160,  # Based on max len of training data + some flexibility
                                  do_sample=True, 
                                  top_k=950, 
                                  repetition_penalty=1.2, 
                                  use_cache=True, 
                                  attention_mask=attention_mask)
        seq = tokenizer.batch_decode(seq_indx, skip_special_tokens=True)

        dict_predictions['mutated_sequence'].extend(seq)

        # Compute perplexity for the sequence
        results = perplexity.compute(model_id=model_path,
                             predictions=seq,
                             add_start_token=False,
                             batch_size=batch_s,   
                             device=dev)
                                    
        dict_predictions["perplexity"].extend(results["perplexities"])
# Save as DataFrame
predictions = pd.DataFrame.from_dict(dict_predictions)
prediction_dir = os.path.join(root_dir,'inference')
os.makedirs(prediction_dir, exist_ok=True)
prediction_file = os.path.join(prediction_dir,f'{model_type}_predictions_trial.csv')
predictions.to_csv(prediction_file, index=False)
