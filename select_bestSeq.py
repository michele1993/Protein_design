import pandas as pd
import numpy as np
import os
from utils import protData_cleaning
from transformers import GPT2Tokenizer

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
dpo_path = os.path.join(root_dir,'inference','dpo_predictions_trial.csv')
base_path = os.path.join(root_dir,'inference','base_predictions_trial.csv')


# Load all data here since lightweight
dpo_seq = pd.read_csv(dpo_path)
base_seq = pd.read_csv(base_path)


print("dpo: ", np.mean(dpo_seq['perplexity']))
print("base: ", np.mean(base_seq['perplexity']))
for d,e in zip(dpo_seq['mutated_sequence'], base_seq['mutated_sequence']):
    print(d==e)
    print(d, "\n")
    print(e, "\n")
    exit()

# Training data
trainingData_path = os.path.join(root_dir,'dataset','sequences.csv')
training_data = pd.read_csv(trainingData_path)
