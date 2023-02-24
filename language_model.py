#%% imports
from process_data import open_json, process_df
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertTokenizerFast, AutoTokenizer

#%%
train_df = open_json("ECHR_Dataset/EN_train")
train_text = train_df["TEXT"]
train_text = process_df(train_text)

train_label = train_df["VIOLATED_ARTICLES"].tolist()
#%%

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

text = x_train[0]
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# encode text
tokens_train = tokenizer(text, truncation=True, max_length=512)

#%%
print(sent_id)

#%%
seq_len = [len(i.split()) for i in text]
# number of sequences 
print(len(seq_len))
# plot histogram to see how long each sequence in the text is
pd.Series(seq_len).hist(bins = 30)
# %%

train_seq = torch.tensor(sent_id['input_ids'])
train_mask = torch.tensor(sent_id['attention_mask'])