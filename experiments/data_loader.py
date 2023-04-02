"""
Import packages
"""

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch

"""
Dictionaries
"""

# label encoding
article_dict = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, '11': 9, '12': 10, '13': 11, '14': 12, '18': 13, '25': 14, '34': 15, '38': 16, '46': 17, 'P1': 18, 'P4': 19, 'P6': 20, 'P7': 21, 'P12': 22}

"""
Functions
"""

def get_one_hot_labels(df, article_dict):
    """
    Create a list of lists with a one-hot encoding of the articles violated for each row in the input dataframe

    Params:
    `df` (pd.DataFrame): dataframe containing the violated articles
    `article_dict` (dict): dictionary mapping article names to indices
    """
    article_dict = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, '11': 9, 
        '12': 10, '13': 11, '14': 12, '18': 13, '25': 14, '34': 15, '38': 16, '46': 17, 'P1': 18, 'P4': 19, 
        'P6': 20, 'P7': 21, 'P12': 22
    }

    labels = []

    for articles in df["violated_articles"]:
        label = [int(key in articles) for key in article_dict.keys()]
        labels.append(label)

    return labels

def load_data(folder="echr", task="binary_cls", anon=False):
    """
    Load data from pickle files and return train, validation and test sets.

    Params:
    `folder` (str): folder containing the data
    `task` (str): task to perform, either 'binary_cls', 'multi_cls' or 'regression'
    `anon` (bool): whether to load the anonymised data or not
    """
    if anon == False:
        train_df = pd.read_pickle(f"data/{folder}/non-anon_train.pkl")
        val_df = pd.read_pickle(f"data/{folder}/non-anon_valid.pkl")
        test_df = pd.read_pickle(f"data/{folder}/non-anon_test.pkl")
    else:
        train_df = pd.read_pickle(f"data/{folder}/anon_train.pkl")
        val_df = pd.read_pickle(f"data/{folder}/anon_valid.pkl")
        test_df = pd.read_pickle(f"data/{folder}/anon_test.pkl")

    if folder == "echr":
        text_column = "text"
    else:
        text_column = "summary"

    train_texts, val_texts, test_texts = train_df[text_column].tolist(), val_df[text_column].tolist(), test_df[text_column].tolist()

    if task == "binary_cls":
        train_labels = train_df["violated"].astype(int).tolist()
        val_labels = val_df["violated"].astype(int).tolist()
        test_labels = test_df["violated"].astype(int).tolist()
    elif task == "multi_cls":
        train_labels = get_one_hot_labels(train_df, article_dict)
        val_labels = get_one_hot_labels(val_df, article_dict)
        test_labels = get_one_hot_labels(test_df, article_dict)
    elif task == "regression":
        train_labels = train_df["importance"].astype(int).tolist()
        val_labels = val_df["importance"].astype(int).tolist()
        test_labels = test_df["importance"].astype(int).tolist()

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def chunking(folder="echr", anon=False):
    
    """
    Generate chunked data for the ECHR dataset.
    
    Params:
    `folder` (str): folder containing the data
    `anon` (bool): whether to load the anonymised data or not
    """
    
    # load data
    if anon == False:
        train_df = pd.read_pickle(f"data/{folder}/non-anon_train.pkl")
        val_df = pd.read_pickle(f"data/{folder}/non-anon_valid.pkl")
        test_df = pd.read_pickle(f"data/{folder}/non-anon_test.pkl")
    else:
        train_df = pd.read_pickle(f"data/{folder}/anon_train.pkl")
        val_df = pd.read_pickle(f"data/{folder}/anon_valid.pkl")
        test_df = pd.read_pickle(f"data/{folder}/anon_test.pkl")

    if folder == "echr":
        text_column = "text"
    else:
        text_column = "summary"
        
    # create new dataframe for each chunk
    new_df = train_df.drop('text', axis=1)
    index_of_text_column = new_df.columns.get_loc(new_df.columns[0])
    new_df.insert(index_of_text_column + 1, 'chunks', '')
    
    # chunk texts and store in new dataframe    
    for i, text in enumerate(train_df["text"].tolist()):
        
        # chunk text
        chunks = chunk_text(text)
        
        # create new rows for each chunk
        row_to_copy = new_df.iloc[i]
        new_rows = pd.concat([row_to_copy]*len(chunks), axis=1).transpose()
        new_df = pd.concat([new_df, new_rows], ignore_index=True)
        
        # store chunks in new dataframe
        for j, chunk in enumerate(chunks):
            new_df['chunks'][len(new_df) - len(chunks) + j] = chunk
            
    # remove first 7100 rows
    new_df = new_df.iloc[7100:]
    
    return new_df
        
def generate_tokens(tokenizer, texts, max_length=512):
    """
    Tokenize the input texts.

    Params:
    `tokenizer` (transformers.PreTrainedTokenizer): tokenizer to use
    `texts` (list): list of texts to tokenize
    `max_length` (int): maximum length of the tokenized texts
    """
    tokens = tokenizer.batch_encode_plus(texts, 
        return_tensors = "pt", 
        padding = "max_length",
        truncation = True, 
        max_length = max_length, 
        pad_to_max_length = True, 
        return_token_type_ids = False
    )

    return tokens

def create_dataloader(tokens, labels, batch_size, type):
    """
    Create a dataloader for the input data.

    Params:
    `tokens` (torch.Tensor): tensor containing the tokenized texts
    `labels` (torch.Tensor): tensor containing the labels
    `batch_size` (int): batch size
    `type` (str): type of dataloader to create, either 'train', 'val' or 'test'
    """
    if not isinstance(labels[0], list):
        labels = torch.tensor(labels).unsqueeze(1)
    else:
        labels = torch.tensor(labels)

    data = TensorDataset(tokens.input_ids, tokens.attention_mask, labels)

    if type == "train":
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    elif type == "val":
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    elif type == "test":
        dataloader = DataLoader(data, batch_size=batch_size)

    return dataloader