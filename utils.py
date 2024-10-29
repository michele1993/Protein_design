import torch
import pandas as pd
import numpy as np
import transformers
import math

def insert_char(seq: str, char: str, every: int) -> list[str]:
    """
    Insert a character into a string every `n` characters.
    """
    return char.join(seq[i:i+every] for i in range(0, len(seq), every)) #.join() all items in a tuple of a string, using char as separator

def protData_cleaning(dataset: pd.DataFrame, remove_activity_NaN: bool=True):
    """ 
    Perform basic data cleaning: remove NaN, duplicates and ODD
    Args:
        dataset: dataset of protein seq and activity in pd.DataFrame, ["mutated_sequence", "activity_dp7"]
        remove_activity_NaN: if True remove entire sequences that have NaN activity (e.g., for RLHF)
    """
    ## ------ 1st Remove  NaN -----
    # NOTE: to train/fine-tune the base model, keep 'normal-looking' sequences with NaN activity
    # since don't need activity

    # identify any NaN sequence
    seq_nan = pd.isna(dataset['mutated_sequence']).astype('bool') # check if there is any NaN sequence

    # remove entries with NaN seq 
    data_cleaned = dataset.loc[~seq_nan, :] # use .loc since bool indexes

    assert ~pd.isna(data_cleaned['mutated_sequence']).any().any(), "There are NaN entries in the sequences, need cleaning"
    
    # When we do RLHF we want to remove sequences with NaN activity
    if remove_activity_NaN:
        activity_nan = pd.isna(data_cleaned['activity_dp7']).astype('bool') # check if there is any NaN sequence
        # remove entries entirely if have NaN activity (e.g., for RLHF)
        data_cleaned = data_cleaned.loc[~activity_nan, :] # use .loc since bool indexes
        assert ~pd.isna(data_cleaned).any().any(), "There are sequences with NaN activity, need cleaning"

    # just in case: reset indexes after above slicing
    data_cleaned = data_cleaned.reset_index(drop=True)

    ## ----------------------------

    ## -------- 2nd Check only known aminoacids are present ----
    # Define natural amino acids
    natural_AM = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L','M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

    # Merge all sequences together
    merged_AM = ''.join(data_cleaned.iloc[:,0])
    # Find unique elements
    unique_AM = set(merged_AM)

    # Assert amino acid is seq is a subset of natural aminoacid
    assert unique_AM.issubset(natural_AM), "The sequences contain some unknown letters, investigate and decide how to deal with them"

    ## ------ 3nd Investigate any duplicate entry ----
    # Check there are not duplicates in the protein seq which reuqire attention
    duplicats = data_cleaned.duplicated(subset="mutated_sequence")
    assert ~duplicats.any().any(), "There are duplicated protein sequences, investigate and decide how to deal with them"

    # 1st option: remove duplicate based on first occurence, may not be best option depending on type of duplicates 
    #data_cleaned = data_cleaned.drop_duplicates(subset="mutated_sequence", keep="first") 
    ## -------------------------------------------

    ## ---- 4rd Remove sequences with out of distribution lengths ---
    seq_lengths = data_cleaned["mutated_sequence"].str.len().tolist()

    # Calculate quartiles and IQR 20-80 %
    Q1 = np.percentile(seq_lengths, 20)
    Q3 = np.percentile(seq_lengths, 80)
    IQR = Q3 - Q1

    # Define bounds with 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # find index of value meeting the boundaries
    indx = [i for i in range(len(seq_lengths)) if lower_bound <= seq_lengths[i] <= upper_bound]

    # select values that meet the boundaries only
    data_cleaned = data_cleaned.iloc[indx,:]

    # reset indexes after above slicing
    data_cleaned = data_cleaned.reset_index(drop=True)

    return data_cleaned

def find_longest_common_prefix(sequence: list[str]) -> str:
    """
    Find the longest common prefix shared by all strings in the input list.

    Args:
        sequence: List of amino acid sequences
    """
    if not sequence:
        return '', 0

    if len(sequence) == 1:
        return sequence[0], len(sequence[0])

    # Find length of shortest string to set upper bound
    min_length = min(len(s) for s in sequence)

    # Compare characters position by position
    common_prefix = []
    for i in range(min_length):
        current_char = sequence[0][i]

        # Check if this character matches in all strings
        if all(s[i] == current_char for s in sequence):
            common_prefix.append(current_char)
        else:
            break

    prefix = ''.join(common_prefix)
    return prefix

def calculatePerplexity(sequence: str, model: transformers.AutoModelForCausalLM, tokenizer: transformers.AutoTokenizer, dev: str):
    """ 
    Compute perplexity for a sequence
    Args:
        sequence: string sequence of amino acids
        model: Head LLM model
        tokenizer: model tokenizer to tokenize the sequence
        dev: device
    """
    input_ids = torch.tensor(tokenizer.encode(sequence),device=dev).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
