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

model_type = 'DPO'

if model_type == 'SFT':
    model_path = os.path.join(root_dir,'output')
elif model_type == 'DPO':
    model_path = os.path.join(root_dir,'dpo_output')
else:
    model_path = 'nferruz/ProtGPT2'

# Initialise fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
ft_protgpt2_generator = pipeline('text-generation', model=model_path, tokenizer=tokenizer, device=dev)

# Propt based on fine-tuning
prompt = "<|endoftext|>ATAPSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPI"

# Generate sequences
n_sequences = 100
with torch.no_grad():
    sequences = ft_protgpt2_generator(prompt, 
                                  max_length=425, 
                                  do_sample=True, 
                                  top_k=950, 
                                  repetition_penalty=1.2, 
                                  num_return_sequences=n_sequences, 
                                  eos_token_id=tokenizer.eos_token_id,
                                  batch_size=1,
                                  use_cache=False
                                 )

# Free-up GPU memory by deleting generator
del ft_protgpt2_generator
torch.cuda.empty_cache()

# Initialise model head to compute perplexity
ft_protgpt2_model = AutoModelForCausalLM.from_pretrained(model_path).to(dev)

# Compute perplexity for each seq and store it with the corresponding seq
dict_predictions = {
        "mutated_sequence": [],
        "activity_dp7": [],
        }
for s in sequences:
    perplexity = calculatePerplexity(sequence=s['generated_text'], model=ft_protgpt2_model, tokenizer=tokenizer, dev=dev)
    seq = s['generated_text'].replace("<|endoftext|>", "")

    dict_predictions["mutated_sequence"] = seq
    dict_predictions["activity_dp7"] = perplexity.to('cpu').item()

# Save as DataFrame
predictions = pd.DataFrame.from_dict(dict_predictions)
prediction_file = os.path.join(root_dir,'predictions.csv')
predictions.to_csv(prediction_file, index=False)
