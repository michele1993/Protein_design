import pandas as pd
import numpy as np

def protData_cleaning(dataset):
    """ 
    Perform basic data cleaning: remove NaN, duplicates and ODD
    Args:
        dataset: dataset of protein seq and activity in pd.DataFrame, ["mutated_sequence", "activity_dp7"]
    """

    ## ------ 1st Remove any pair if contains NaN -----
    # identify any row with NaN
    data_nan = pd.isna(dataset).sum(axis=1).astype('bool') # sum to identify if there is at least one NaN entry in a row.
    # remove entries with NaN entirely 
    data_cleaned = dataset.loc[~data_nan, :] # use .loc since bool indexes

    # just in case: reset indexes after above slicing
    data_cleaned = data_cleaned.reset_index(drop=True)

    assert ~pd.isna(data_cleaned).any().any(), "There are NaN entries in the data, need cleaning"
    ## ----------------------------

    ## ------ 2nd Investigate any duplicate entry ----
    # Check there are not duplicates in the protein seq which reuqire attention
    duplicats = data_cleaned.duplicated(subset="mutated_sequence")
    assert ~duplicats.any().any(), "There are duplicated protein sequences, investigate and decide how to deal with them"

    # 1st option: remove duplicate based on first occurence, may not be best option depending on type of duplicates 
    #data_cleaned = data_cleaned.drop_duplicates(subset="mutated_sequence", keep="first") 
    ## -------------------------------------------

    ## ---- 3rd Remove sequences with out of distribution lengths ---
    seq_lengths = data_cleaned["mutated_sequence"].str.len().tolist()

    # Calculate quartiles and IQR
    #NOTE: there are a few very short-lenght seq and a couple 
    # of longer-length seq, whcih are close to mode length
    # so decided to keep upper percentile to compute IQR.
    Q1 = np.percentile(seq_lengths, 20)
    Q3 = np.percentile(seq_lengths, 100)
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

