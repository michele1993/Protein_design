import pandas as pd
import numpy as np

def insert_char(seq, char, every):
    """
    Insert a character into a string every `n` characters.
    """
    return char.join(seq[i:i+every] for i in range(0, len(seq), every))

def protData_cleaning(dataset, remove_activity_NaN=True):
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

