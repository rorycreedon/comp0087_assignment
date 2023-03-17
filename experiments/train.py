"""
import packages
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import time
import os
from config import device
import argparse
from data_loader import load_data, generate_tokens, create_dataloader
from get_loss import get_loss_func
import logging
from model import LegalBertBinaryCls, LegalBertMultiCls, LegalBertRegression

"""
functions
"""

# train the model
def train(train_loader, model, task, lr):
    start = time.time()

    optimizer = AdamW(model.parameters(), lr=lr) 

    loss_func = get_loss_func(task)

    model.train()

    # iterate over batches
    for i, batch in enumerate(train_loader):
        # initialise
        running_loss = 0

        # progress update after every 100 batches.
        if i % 100 == 0:
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

def validate(val_loader, model, task):
    start = time.time()
    running_loss = 0

    loss_func = get_loss_func(task)

    model.eval()

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

    print(f"----> validation loss {running_loss}")
    print(f"----> time taken {time.time()-start}")
    print()

    return running_loss

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
    parser.add_argument("--anon", type=bool, default=False)
    parser.add_argument("--tasks", type=tuple, default = ("binary_cls", "regression"), help = "The tasks to be worked on")
    parser.add_argument("--model_name", type=str, default = "legalbert_echr_truncation-1", help = "The path to save the model")
    parser.add_argument("--max_seq_length", type=int, default = 512, help = "The maximum length of the input sequence")
    parser.add_argument("--num_epochs", type=int, default = 6, help = "Number of epochs")
    parser.add_argument("--lr", type=float, default = 3e-5, help = "Learning rate")
    parser.add_argument("--batch_size",type = int, default = 4, help = "Batch size")
    parser.add_argument("--pretrained_model", type=str, default = "nlpaueb/legal-bert-small-uncased", help = "The path to the pretrained model")

    args = parser.parse_args()

    # create a logger object
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    log_dir = "experiments/logs/train"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, args.model_name + ".log"))
    fh.setLevel(logging.INFO)
    # create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the file handler to the logger
    logger.addHandler(fh)

    # save information for this task
    logger.info('folder: %s', args.folder)
    logger.info('anon: %s', args.anon)
    logger.info('model name: %s', args.model_name)
    logger.info('max sequence length: %d', args.max_seq_length)
    logger.info('number of epochs: %d', args.num_epochs)
    logger.info('learning rate: %f', args.lr)
    logger.info('batch size: %d', args.batch_size)
    logger.info('pretrained model: %s', args.pretrained_model)

    # Make folders if necessary
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('models/binary_cls'):
        os.mkdir('models/binary_cls')
    if not os.path.exists('models/multi_cls'):
        os.mkdir('models/multi_cls')
    if not os.path.exists('models/regression'):
        os.mkdir('models/regression')

    # use gpu
    print(f"using {device}")
    print()

    print(f"folder: {args.folder}")
    print()

    for task in args.tasks:
        print("task: ", task)
        print()

        # binary classification on non-anon echr data
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data(args.folder, task, args.anon)

        # remove warning
        transformers.logging.set_verbosity_error()

        # load model
        model = AutoModel.from_pretrained(args.pretrained_model, return_dict=False)
        # adapt model
        model = LegalBertBinaryCls(model)
        # move model to gpu
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

        train_tokens = generate_tokens(tokenizer, train_texts, args.max_seq_length)
        val_tokens = generate_tokens(tokenizer, val_texts, args.max_seq_length)

        train_loader = create_dataloader(train_tokens, train_labels, args.batch_size, type="train")
        val_loader = create_dataloader(val_tokens, val_labels, args.batch_size, type="val")

        train_losses = []
        val_losses = []
        
        
        for epoch in range(args.num_epochs):
            print('epoch {:} / {:}'.format(epoch + 1, args.num_epochs))

            #train model
            train_loss, _ = train(train_loader, model, task, args.lr)
            #evaluate model
            val_loss, _ = validate(val_loader, model, task)

            # append training and validation loss
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        
        # save trained model
        torch.save(model.state_dict(), f'models/{str(task)}/{str(args.model_name)}.pt')
        print('model saved')
        print()

        # save information for this task
        logger.info('[%s] Training loss: %s', task, str(train_losses))
        logger.info('[%s] Validation losses: %s', task, str(val_losses))

    print("script finishes")
    print("=========================")


        
