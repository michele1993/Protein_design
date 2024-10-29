import torch
from transformers import pipeline, GPT2Tokenizer, AutoModelForCausalLM 
import os
from torch.distributions import Categorical
import torch.nn.functional as F
from utils import calculatePerplexity
import pandas as pd

if torch.cuda.is_available():
    dev='cuda'
else:
    dev='cpu'

# Get path to fine-tuned model
root_dir = os.path.dirname(os.path.abspath(__file__))

model_type = 'dpo'

if model_type == 'sft':
    model_path = os.path.join(root_dir,'output')
elif model_type == 'dpo':
    model_path = os.path.join(root_dir,'dpo_output')
else:
    model_path = 'nferruz/ProtGPT2'

# Initialise fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
#ft_protgpt2_generator = pipeline('text-generation', model=model_path, tokenizer=tokenizer, device=dev)

# Propt based on fine-tuning
prompt = "<|endoftext|>ATAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPI"

# Generate sequences
n_sequences = 5
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

# Initialise model head to compute perplexity
inputs = tokenizer(prompt, return_tensors="pt").input_ids
ft_protgpt2_model = AutoModelForCausalLM.from_pretrained(model_path).to(dev)
outputs = model.generate(inputs, max_new_tokens=425, do_sample=True, top_k=950, top_p=0.95)
print(outputs.shape)
exit()

# Compute perplexity for each seq and store it with the corresponding seq
dict_predictions = {
        "mutated_sequence": [],
        "perplexity": [],
        }
batch_s = 5
for i in range(0,len(sequences), batch_s):
    s = sequences[i:i+batch_s]
    perplexity = calculatePerplexity(sequence=s, model=ft_protgpt2_model, tokenizer=tokenizer, dev=dev)
    seq = s['generated_text'].replace("<|endoftext|>", "")

    dict_predictions["mutated_sequence"].append(seq)
    dict_predictions["perplexity"].append(perplexity)

# Save as DataFrame
predictions = pd.DataFrame.from_dict(dict_predictions)
prediction_dir = os.path.join(root_dir,'inference')
os.makedirs(prediction_dir, exist_ok=True)
prediction_file = os.path.join(prediction_dir,f'{model_type}_predictions.csv')
predictions.to_csv(prediction_file, index=False)
