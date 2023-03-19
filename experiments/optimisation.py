"""
import packages
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error
import time
from data_loader import load_data, generate_tokens, create_dataloader
from train import train
from hyperopt import fmin, tpe, hp, Trials
from config import device
from get_loss import get_loss_func
import logging
import argparse
import transformers
from model import get_model
import os
from functools import partial
import hyperopt
import pickle

"""
functions
"""

def validate_withscore(val_loader, model, task):
    start = time.time()
    running_loss = 0
    acc = 0

    loss_func = get_loss_func(task)

    model.eval()

    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

    # iterate over batches
    for i, batch in enumerate(val_loader):
        # progress update after every 100 batches.
        if i % 100 == 0:
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
    
    running_loss = running_loss / len(val_loader)

    if task == "binary_cls" or task == "multi_cls":
        score = f1_score(all_labels, all_preds)
    elif task == "regression":
        score = mean_absolute_error(all_labels, all_preds)

    print(f"----> validation loss {running_loss}")
    print(f"----> validation f1 score {score}")
    print(f"----> time taken {time.time()-start}")
    print()

    return running_loss, score

# Define the objective function to optimize
def objective(params, folder, task, pretrained_model, max_length, num_epochs):
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data(folder, task)

    # remove warning
    transformers.logging.set_verbosity_error()

    # load model
    model = AutoModel.from_pretrained(args.pretrained_model, return_dict=False)
    # adapt model
    model = get_model(task, model)
    # move model to gpu
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    train_tokens = generate_tokens(tokenizer, train_texts, max_length)
    val_tokens = generate_tokens(tokenizer, val_texts, max_length)

    train_loader = create_dataloader(train_tokens, train_labels, params["batch_size"], type="train")
    val_loader = create_dataloader(val_tokens, val_labels, params["batch_size"], type="val")

    # move model to gpu
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        #train model
        train_loss = train(train_loader, model, task, params["lr"])
        #evaluate model
        val_loss, score = validate_withscore(val_loader, model, task)
    
        # append training and validation loss
        train_losses.append(train_loss)
        val_losses.append(val_loss)

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

        # parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default="echr")
    parser.add_argument("--tasks", type=list, default = ["binary_cls"], help = "The tasks to be worked on")
    parser.add_argument("--file_name", type=str, default = "echr", help = "The path to the log file")
    parser.add_argument("--max_seq_length", type=int, default = 512, help = "The maximum length of the input sequence")
    parser.add_argument("--num_epochs", type=int, default = 4, help = "Maximum number of epochs")
    parser.add_argument("--lr", type=list, default = [2e-5, 3e-5, 4e-5, 5e-5], help = "Learning rate range")
    parser.add_argument("--batch_size",type=list, default = [4, 8, 16], help = "Batch size range")
    parser.add_argument("--pretrained_model", type=str, default = "nlpaueb/legal-bert-small-uncased", help = "The path to the pretrained model")
    parser.add_argument("--num_trials", type=int, default = 1, help = "Number of trials")

    args = parser.parse_args()

    # use gpu
    print(f"using {device}")
    print()

    print(f"folder: {args.folder}")
    print()

    print(f"We will be evaluating the following tasks: {args.tasks}")
    print()
    
    for task in args.tasks:
        print("task: ", task)
        print()
        
        fmin_objective = partial(objective, 
                                 folder=args.folder, 
                                 task=task, 
                                 pretrained_model=args.pretrained_model, 
                                 max_length=args.max_seq_length,
                                 num_epochs=args.num_epochs
                                 )

        # search space
        space = {
            'lr': hp.choice('lr', args.lr),
            'batch_size': hp.choice('batch_size', args.batch_size),
        }   

        # create a Trials object to store the results
        trials = Trials()

        best = fmin(fn=fmin_objective, 
                    space=space, 
                    algo=tpe.suggest, 
                    max_evals=args.num_trials,
                    trials = trials,
                    )

        # print the best hyperparameters
        print(f"best hyperparameters {best}")

        log_dir = "experiments/logs/opt/" + str(task)
        os.makedirs(log_dir, exist_ok=True)

        info = {
            'folder': args.folder,
            'max_seq_length': args.max_seq_length,
            'pretrained_model': args.pretrained_model,
            'learning_rate_range': args.lr,
            'batch_size_range': args.batch_size,
            'num_epochs': args.num_epochs,
            'num_trials': args.num_trials,
            'trials': trials
        }

        # save file
        with open(log_dir + args.file_name + '.pkl', 'wb') as f:
            pickle.dump(info, f)

    print("script finishes")
    print("=========================")