import os
import pandas as pd

# Load data
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir,'dataset','sequences.csv')

dataset = pd.read_csv(file_path)

## ------ 1st) Remove any pair if contains NaN -----
# identify any row with NaN
data_nan = pd.isna(dataset).sum(axis=1).astype('bool') # sum to identify if there is at least one NaN entry in a row.
# remove entries with NaN entirely 
data = dataset.loc[~data_nan, :]

assert ~pd.isna(data).any().any(), "There are NaN entries in the data, need cleaning"
## ----------------------------

# Use .iloc[] to select data based on integer positions (like indexing in a list)
#print(dataset.iloc[0,0])

for i in range(data.shape[0]):
    #print(len(dataset.iloc[i,0]))
    print(data.iloc[i,0])

#print(dataset[0])
print(type(dataset))

print(data.shape)
