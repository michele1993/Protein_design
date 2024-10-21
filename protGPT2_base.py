from transformers import pipeline

protgpt2 = pipeline('text-generation', model="nferruz/ProtGPT2")

trial_seq = 'KKKKK'


input_ids = protgpt2.tokenizer.encode(trial_seq)
print(input_ids)
print(len(trial_seq))
sequences = protgpt2("<|endoftext|>", max_length=100, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=10, eos_token_id=0)

for s in sequences:
    print(s)
