"""
import packages
"""

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch

"""
functions
"""

def load_data(folder="echr", task="binary_cls", anon=False):
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
    elif task == "regression":
        train_labels = train_df["importance"].astype(int).tolist()
        val_labels = val_df["importance"].astype(int).tolist()
        test_labels = test_df["importance"].astype(int).tolist()

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

# tokenize texts
def generate_tokens(tokenizer, texts, max_length=512):
    tokens = tokenizer.batch_encode_plus(texts, 
        return_tensors = "pt", 
        padding = "max_length",
        truncation = True, 
        max_length = max_length, 
        pad_to_max_length = True, 
        return_token_type_ids = False
    )

    return tokens

# create dataloaders
def create_dataloader(tokens, labels, batch_size, type):
    data = TensorDataset(tokens.input_ids, tokens.attention_mask, torch.tensor(labels))

    if type == "train":
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    elif type == "val":
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    elif type == "test":
        dataloader = DataLoader(data, batch_size=batch_size)

    return dataloader