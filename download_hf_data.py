"""
Import packages
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import os

"""
Functions
"""

def clean_echr(df):
    """
    Clean ECHR dataset.

    Params:
    `df` (pd.DataFrame): dataframe to clean
    """

    # Drop rows where language is not English
    if (np.unique(df['languageisocode']) == 'ENG')!=True:
        df = df[df['languageisocode'] == 'ENG']

    # Drop rows where text is empty
    df = df[df['text'].apply(lambda x: len(x)) > 0]

    # Convert array of strings to string in the 'text' column
    df['text'] = df['text'].apply(lambda x: ' '.join(x))

    # Dummy variable for when conclusion = 'Inadmissible'
    df['inadmissible'] = df['conclusion'] == 'Inadmissible'

    # Create column for articles raised
    df['all_articles'] = df['violated_articles'] + df['non_violated_articles'] 

    # Keep only columns of interest
    df = df[['itemid', 'text', 'violated_articles', 'violated', 'non_violated_articles', 'all_articles', 'importance', 'inadmissible', 'date', 'docname']]

    return df


def download_echr(name):
    """
    Download and clean ECHR dataset from Hugging Face datasets library.

    Params:
    `name` (str): name of dataset to download, either 'anon' or 'non-anon'
    """

    # Check if name is valid
    if name not in ['anon', 'non-anon']:
        raise ValueError("Name must be either 'anon' or 'non-anon'")

    # Download dataset
    data = load_dataset(path = "jonathanli/echr", name = name)

    # Convert test, train and validation to dataframes
    test_df = pd.DataFrame(data['test'])
    train_df = pd.DataFrame(data['train'])
    valid_df = pd.DataFrame(data['validation'])

    # Clean each dataframe
    test_df = clean_echr(test_df)
    train_df = clean_echr(train_df)
    valid_df = clean_echr(valid_df)

    # Make data and echr folder if necessary
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/echr'):
        os.mkdir('data/echr')

    # Save each dataframe to pickle file
    test_df.to_pickle(f'data/echr/{name}_test.pkl')
    train_df.to_pickle(f'data/echr/{name}_train.pkl')
    valid_df.to_pickle(f'data/echr/{name}_valid.pkl')

"""
Run code
"""

if __name__ == '__main__':

    # Download and clean ECHR datasets
    download_echr('anon')
    download_echr('non-anon')