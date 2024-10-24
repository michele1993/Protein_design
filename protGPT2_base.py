from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
import os

model_name = "nferruz/ProtGPT2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model_head = GPT2LMHeadModel.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

#trial_seq = '<|endoftext|>KKKKKKKKKKKKK<|endoftext|>'
#input_ids = tokenizer.encode(trial_seq)
#input_ids = tokenizer(trial_seq, return_tensors='pt')
#outputs = model(**input_ids)
#print(outputs[0].shape)
#exit()


# Get path to fine-tuned model
root_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root_dir,'output')

protgpt2 = pipeline('text-generation', model=model_path)
#dev = torch.device("cpu")

# Load tokenizer and model
#tokenizer = protgpt2.tokenizer.from_pretrained(model_path)
#ft_protgpt2 = protgpt2.from_pretrained(model_path).to(dev)

# Generate sequence
sequences = ft_protgpt2("<|endoftext|>", max_length=425, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)


#trial_seq = 'KKKKK'
#input_ids = protgpt2.tokenizer.encode(trial_seq)
#sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)

for s in sequences:
    print(s)
