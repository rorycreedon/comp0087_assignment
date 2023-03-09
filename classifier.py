"""
import packages
"""

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging, AutoModel
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
import time

"""
functions
"""

# load the original echr data
def load_echr(task="binary_cls", anon=False):
    if anon == False:
        train_df = pd.read_pickle("data/echr/non-anon_train.pkl")
        val_df = pd.read_pickle("data/echr/non-anon_valid.pkl")
        test_df = pd.read_pickle("data/echr/non-anon_test.pkl")
    else:
        train_df = pd.read_pickle("data/echr/anon_train.pkl")
        val_df = pd.read_pickle("data/echr/anon_valid.pkl")
        test_df = pd.read_pickle("data/echr/anon_test.pkl")

    if task == "binary_cls":
        train_texts, train_labels = train_df["text"].tolist(), train_df["violated"].astype(int).tolist()
        val_texts, val_labels = val_df["text"].tolist(), val_df["violated"].astype(int).tolist()
        test_texts, test_labels = test_df["text"].tolist(), test_df["violated"].astype(int).tolist()

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
def create_dataloader(tokens, labels, type, batch_size=32):
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

def train(train_loader, model, task="binary_cls", lr=1e-3, weight_decay=1e-3):
    start = time.time()
    running_loss = 0

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 

    if task == "binary_cls":
        loss_func = nn.BCEWithLogitsLoss()

    model.train()

    # iterate over batches
    for i, batch in enumerate(train_loader):
        # progress update after every 50 batches.
        if i % 50 == 0:
            print("--> batch {:} of {:}.".format(i, len(train_loader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()

        # forward pass  
        preds = model(input_ids, attention_mask)
        preds, labels = preds.type(torch.FloatTensor), labels.type(torch.FloatTensor)

        # compute the loss between actual and predicted values
        loss = loss_func(preds, labels)
        # add on to the total loss
        running_loss += loss.item()
        # backward pass to calculate the gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # update parameters
        optimizer.step()
    
    print(f"----> training loss {running_loss}")
    print(f"----> time taken {time.time()-start}")

    return running_loss

def validate(val_loader, model, task="binary_cls"):
    start = time.time()
    running_loss = 0

    if task == "binary_cls":
        loss_func = nn.BCEWithLogitsLoss()

    model.eval()

    # iterate over batches
    for i, batch in enumerate(val_loader):
        # progress update after every 50 batches.
        if i % 50 == 0:
            print("--> batch {:} of {:}.".format(i, len(val_loader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            # forward pass  
            preds = model(input_ids, attention_mask)
            preds, labels = preds.type(torch.FloatTensor), labels.type(torch.FloatTensor)

            # compute the loss between actual and predicted values
            loss = loss_func(preds, labels)
            # add on to the total loss
            running_loss += loss.item()

    print(f"----> validation loss {running_loss}")
    print(f"----> time taken {time.time()-start}")
    print()

    return running_loss

def test(test_loader, model, model_name):
    # loaded trained model
    model.load_state_dict(torch.load(f'models/{str(model_name)}.pt'))

    all_preds = []
    all_labels = []

    # iterate over batches
    for i, batch in enumerate(test_loader):
        # progress update after every 50 batches.
        if i % 50 == 0:
            print("--> batch {:} of {:}.".format(i, len(test_loader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            # forward pass  
            preds = model(input_ids, attention_mask)
            preds, labels = preds.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            preds = np.argmax(preds, axis = 1)
            all_preds.append(preds)

    print(classification_report(all_labels, all_preds))
    print()

    return

"""
classes
"""

class BERT(nn.Module):
    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert 

    def forward(self, input_ids, attention_mask):
        preds = self.bert(input_ids, attention_mask, return_dict=False)

        return preds[0][:,1]
    
"""
tasks
"""

if __name__ == "__main__":
    print("=========================")
    print("script starts...")
    print()

    # use gpu
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using {device}")
    print()

    task = "binary_cls"
    model_name = "bert-1" 
    max_length = 5 # !!!
    print(f"task {task}")
    print(f"model name {model_name}")
    print(f"max sequence length {max_length}")
    print()
    
    # hyperparameters
    print("hyperparameters")
    num_epochs = 2
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 32
    print(f"--> number of epochs {num_epochs}")
    print(f"--> learning rate {lr}")
    print(f"--> weight decay {weight_decay}")
    print(f"--> batch size {batch_size}")
    print()

    # binary classification on non-anon echr data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_echr(task="binary_cls", anon=False)

    # remove warning
    logging.set_verbosity_error()

    # use bert
    pretrained_model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, return_dict=False)
    # adapt bert
    model = BERT(model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    train_tokens = generate_tokens(tokenizer, train_texts, max_length)
    val_tokens = generate_tokens(tokenizer, val_texts, max_length)
    test_tokens = generate_tokens(tokenizer, test_texts, max_length)

    train_loader = create_dataloader(train_tokens, train_labels, batch_size, type="train")
    val_loader = create_dataloader(val_tokens, val_labels, batch_size, type="val")
    test_loader = create_dataloader(test_tokens, test_labels, batch_size, type="test")

    # move model to gpu
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print('epoch {:} / {:}'.format(epoch + 1, num_epochs))

        #train model
        train_loss = train(train_loader, model, task, lr, weight_decay)
        #evaluate model
        val_loss = validate(val_loader, model, task)

        # append training and validation loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # save trained model
    torch.save(model.state_dict(), f'models/{str(task)}/{str(model_name)}.pt')
    print('model saved')
    print()

    # test
    test(test_loader, model, model_name)

    print("script finishes")
    print("=========================")