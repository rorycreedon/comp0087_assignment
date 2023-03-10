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
import os
from config import device

"""
functions
"""

# load the original echr data
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

    if task == "binary_cls":
        train_texts, train_labels = train_df[text_column].tolist(), train_df["violated"].astype(int).tolist()
        val_texts, val_labels = val_df[text_column].tolist(), val_df["violated"].astype(int).tolist()
        test_texts, test_labels = test_df[text_column].tolist(), test_df["violated"].astype(int).tolist()

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

def train(train_loader, model, task="binary_cls", lr=1e-3, weight_decay=1e-3):
    start = time.time()
    running_loss = 0
    acc = 0

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

        if task == "binary_cls":
                probs = nn.Sigmoid()(preds)
                preds = (probs > 0.5).long()
                acc += torch.sum(preds == labels)

    acc = acc / (len(train_loader)*len(batch))
    
    print(f"----> training loss {running_loss}")
    print(f"----> training accuracy {acc}")
    print(f"----> time taken {time.time()-start}")

    return running_loss

def validate(val_loader, model, task="binary_cls"):
    start = time.time()
    running_loss = 0
    acc = 0

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

            if task == "binary_cls":
                probs = nn.Sigmoid()(preds)
                preds = (probs > 0.5).long()
                acc += torch.sum(preds == labels)

    acc = acc / (len(val_loader)*len(batch))

    print(f"----> validation loss {running_loss}")
    print(f"----> validation accuracy {acc}")
    print(f"----> time taken {time.time()-start}")
    print()

    return running_loss

def test(test_loader, model, task, model_name=""):
    if model_name != "":
        # loaded trained model
        model.load_state_dict(torch.load(f'models/{str(task)}/{str(model_name)}.pt'))

    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

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
            if task == "binary_cls":
                probs = nn.Sigmoid()(preds)
                preds = (probs > 0.5).long()
            #preds = np.argmax(preds, axis=1)
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    print(classification_report(all_labels, all_preds))
    print()

    return all_preds, all_labels

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

    # Make folders if necessary
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('models/binary_cls'):
        os.mkdir('models/binary_cls')

    # use gpu
    print(f"using {device}")
    print()

    folder = "long_t5_summary"
    task = "binary_cls"
    model_name = "bert-1" 
    max_length = 512 # !!!
    print(f"folder {folder}")
    print(f"task {task}")
    print(f"model name {model_name}")
    print(f"max sequence length {max_length}")
    print()
    
    # hyperparameters
    print("hyperparameters")
    num_epochs = 4
    lr = 1e-3
    weight_decay = 1e-3
    batch_size = 8
    print(f"--> number of epochs {num_epochs}")
    print(f"--> learning rate {lr}")
    print(f"--> weight decay {weight_decay}")
    print(f"--> batch size {batch_size}")
    print()

    # binary classification on non-anon echr data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data(folder, task, anon=False)

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
    """
    for epoch in range(num_epochs):
        print('epoch {:} / {:}'.format(epoch + 1, num_epochs))

        #train model
        train_loss, _ = train(train_loader, model, task, lr, weight_decay)
        #evaluate model
        val_loss, _ = validate(val_loader, model, task)

        # append training and validation loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # save trained model
    torch.save(model.state_dict(), f'models/{str(task)}/{str(model_name)}.pt')
    print('model saved')
    print()
    """
    # test
    #test(test_loader, model, model_name, task)
    test_preds, test_labels = test(test_loader, model, task, model_name="")

    print("script finishes")
    print("=========================")