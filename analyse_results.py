import pandas as pd
import numpy as np
import os
from utils import protData_cleaning
from transformers import GPT2Tokenizer
import json

root_dir = os.path.dirname(os.path.abspath(__file__))
result_dir = os.path.join(root_dir,'results')

# Compare perplexity for best 10% activity training sequences between base and dpo-fine-tuned protGPT2  
training_perplexities_file = os.path.join(result_dir,'TrainingDataPerplexities.json')
training_perplexities = json.load(open(training_perplexities_file))

print("base: ",training_perplexities['base_perplexity']) 
print("sft: ",training_perplexities['sft_perplexity']) 
print("dpo: ",training_perplexities['dpo_perplexity'], "\n") 
exit()

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
dpo_path = os.path.join(result_dir,'dpo_predictions_trial.csv')
base_path = os.path.join(result_dir,'base_predictions_trial.csv')


# Load all data here since lightweight
dpo_seq = pd.read_csv(dpo_path)
base_seq = pd.read_csv(base_path)


print("dpo: ", np.mean(dpo_seq['perplexity']))
print("base: ", np.mean(base_seq['perplexity']))
exit()
for d,e in zip(dpo_seq['mutated_sequence'], base_seq['mutated_sequence']):
    print(d==e)
    print(d, "\n")
    print(e, "\n")
    exit()

# Training data
trainingData_path = os.path.join(root_dir,'dataset','sequences.csv')
training_data = pd.read_csv(trainingData_path)
