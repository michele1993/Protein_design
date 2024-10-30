import pandas as pd
import numpy as np
import os
from utils import protData_cleaning
from transformers import GPT2Tokenizer
import json
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.abspath(__file__))

# Training data
trainingData_path = os.path.join(root_dir,'dataset','sequences.csv')
training_data = pd.read_csv(trainingData_path)

training_data = protData_cleaning(dataset=training_data, remove_activity_NaN=True)
x = training_data['activity_dp7'].values

# Compare dpo-FT model perplexity across training data
result_dir = os.path.join(root_dir,'results')
training_perplexities_file = os.path.join(result_dir,'TrainingDataPerplexities.json')
training_perplexities = json.load(open(training_perplexities_file))

y = training_perplexities['dpo_perplexity'] #NOTE: ensured the training data perplexities order reflects the original order

plt.figure(figsize=(6, 6))
plt.scatter(x, y, color='tab:blue',alpha=0.8)
plt.xlabel("actvity_dp7")
plt.ylabel("Perxplexity")
plt.title("dpo-FT protGPT2 perplexity vs activity")
ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

print(len(dpo_training_perplexities))
print(len(activities))
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

