import pandas as pd
import numpy as np
import os
from utils import protData_cleaning
from transformers import GPT2Tokenizer
import json
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.abspath(__file__))
save_figs = True
save_seq = False

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
## ---- Plot perplexity of DPO training data relative to activity ----
plt.figure(figsize=(6, 6))
plt.scatter(x, y, color='tab:blue',alpha=0.8)
plt.xlabel("actvity_dp7")
plt.ylabel("Perplexity")
plt.title("dpo-FT protGPT2 perplexity vs activity")
ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.show()
fig_dir = os.path.join(root_dir,'img')
os.makedirs(fig_dir, exist_ok=True)
fig_file = os.path.join(fig_dir,'perplexity_vs_activity.png')
if save_figs:
    plt.savefig(fig_file, format='png', dpi=1400)

## ------ Plot SFT vs base model perplexity for training data ----
conditions = ['base-model', 'SFT-model']
# Bar width and positions
index = np.arange(len(conditions))
values = [np.mean(training_perplexities['base_perplexity']), np.mean(training_perplexities['sft_perplexity'])]
# Plotting the bars
plt.figure(figsize=(6, 6))
plt.bar(index, values, color= ['tab:orange','tab:green'], alpha=0.8, width = 0.50)
fig_file = os.path.join(fig_dir,'base_vs_sft.png')

# Labels and title
plt.xlabel('Conditions')
plt.ylabel('Perplexity')
plt.title('Base vs SFT model perplexity on training data')
ax = plt.gca()  # Get current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(index, conditions)  # Centering the x-ticks
if save_figs:
    plt.savefig(fig_file, format='png', dpi=1400)


# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
dpo_path = os.path.join(result_dir,'dpo_predictions.csv')


# Find best generated sequenses using a mix of perplexity values and seq lenght
dpo_seq = pd.read_csv(dpo_path)
mean_perplexity = dpo_seq['perplexity'].mean()
std_perplexity = dpo_seq['perplexity'].std()
dpo_sorted = dpo_seq.sort_values(by='perplexity')
dpo_sorted.reset_index(drop=True)


best_seq = []
perplexities = []
for s,p in zip(dpo_sorted['mutated_sequence'], dpo_sorted['perplexity']):
    clean_seq = "".join(char for char in s if char.isalpha())
    if len(clean_seq) <= 430 and len(clean_seq) >= 420:
        best_seq.append(clean_seq)
        perplexities.append(p)
    if len(best_seq) == 100:
        break

# Write to file
if save_seq:
    with open("results/best_100_seq.txt", "w") as file:
        for s in best_seq:
            file.write(s + "\n")
