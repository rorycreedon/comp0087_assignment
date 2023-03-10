"""
import packages
"""

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging, AutoModel
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
import time
import os
from classifier import load_echr, generate_tokens, create_dataloader, train, BERT
from hyperopt import fmin, tpe, hp
from sklearn.metrics import f1_score
from config import device

"""
functions
"""

def validate_withscore(val_loader, model, task="binary_cls"):
    start = time.time()
    running_loss = 0
    acc = 0

    if task == "binary_cls":
        loss_func = nn.BCEWithLogitsLoss()

    model.eval()

    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

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

            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    acc = acc / (len(val_loader)*len(batch))
    score = f1_score(all_labels, all_preds)

    print(f"----> validation loss {running_loss}")
    print(f"----> validation accuracy {acc}")
    print(f"----> validation f1 score {score}")
    print(f"----> time taken {time.time()-start}")
    print()

    return running_loss, score


# Define the objective function to optimize
def objective(params, pretrained_model_name="bert-base-uncased", num_epochs=5):
    # binary classification on non-anon echr data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_echr(task="binary_cls", anon=False)

    # remove warning
    logging.set_verbosity_error()
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, return_dict=False)
    # adapt bert
    model = BERT(model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    train_tokens = generate_tokens(tokenizer, train_texts, max_length)
    val_tokens = generate_tokens(tokenizer, val_texts, max_length)

    train_loader = create_dataloader(train_tokens, train_labels, params["batch_size"], type="train")
    val_loader = create_dataloader(val_tokens, val_labels, params["batch_size"], type="val")

    # move model to gpu
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print('epoch {:} / {:}'.format(epoch + 1, num_epochs))

        #train model
        train_loss = train(train_loader, model, task, params["lr"], params["weight_decay"])
        #evaluate model
        val_loss, score = validate_withscore(val_loader, model, task)
    
        # append training and validation loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Return the negative validation accuracy (to maximize accuracy)
    return -score
    
"""
classes
"""
    
"""
tasks
"""

if __name__ == "__main__":
    print("=========================")
    print("script starts...")
    print()

    # Make folders if necessary

    # use gpu
    print(f"using {device}")
    print()

    task = "binary_cls"
    model_name = "bert_opt-1" 
    max_length = 5 # !!!
    print(f"task {task}")
    print(f"max sequence length {max_length}")
    print()
    
    # search space
    space = {
    'lr': hp.choice('lr', 1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
    'batch_size': hp.choice('batch_size', [4, 8, 16, 32]),
    'weight_decay': hp.loguniform('weight_decay', -3, -1),
    }   

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)
    print('Best hyperparameters:', best)

    print("script finishes")
    print("=========================")