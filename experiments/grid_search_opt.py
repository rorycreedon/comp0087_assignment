"""A simple Grid Search Optimisation for Pre-Analysis Results for LegalBert."""

# Standard Library
import time
import os
from config import device
import argparse
import logging
# Third Party Library
from sklearn.metrics import f1_score, mean_absolute_error
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from data_loader import load_data, generate_tokens, create_dataloader
from model import LegalBertBinaryCls, LegalBertMultiCls, LegalBertRegression
# Private Library
from model import get_model

class GridSearchOpt():

    def __init__(self, args, logger):
        self.args = args
        self.logger  = logger

    def log_results(self, task, lr, batch_size, train_losses, val_losses, metric_values):
        # Log the results and hyperparameters for this hyperparameter combination
        self.logger.info("[%s] Hyperparameters: LR=%f, Batch Size=%d", task, lr, batch_size)
        self.logger.info("[%s] Training losses: %s", task, str(train_losses))
        self.logger.info("[%s] Validation losses: %s", task, str(val_losses))
        self.logger.info("[%s] Metric values: %s", task, str(metric_values))

    def train(self, model, train_loader, lr, task):
        # Start timer
        start = time.time()
        # Initialise running loss
        running_loss = 0
        # Initialise optimiser
        optimizer = AdamW(model.parameters(), lr=lr) 
        # Obtain loss function for specific task
        loss_func = self.get_loss_func(task)
        # Begin training phase
        model.train()
        # Iterate over batches
        for i, batch in enumerate(train_loader):
            # Progress update after every 100 batches.
            if i % 100 == 0:
                print("--> batch {:} of {:}.".format(i, len(train_loader)))
            # push the batch to gpu
            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch
            # Zero gradients to prevent accumulation
            optimizer.zero_grad()
            # Forward pass  
            preds = model(input_ids, attention_mask)
            # Convert to tensor
            preds, labels = preds.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            # Compute the loss between actual and predicted values
            loss = loss_func(preds, labels)
            # Add on to the total loss
            running_loss += loss.item()
            # Backward pass to calculate the gradients
            loss.backward()
            # Update parameters
            optimizer.step()
        # Calculate running loss
        running_loss /= len(train_loader)
        print(f"----> Training loss {running_loss}")
        print(f"----> Time taken {time.time()-start}")
        return running_loss

    def validate(self, model, val_loader, task):
        # Start timer
        start = time.time()
        # Initiliase running loss
        running_loss = 0
         # Obtain loss function for specific task
        loss_func = self.get_loss_func(task)
        # Begin evaluation phase
        model.eval()
        # Store results
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        # Iterate over batches
        for i, batch in enumerate(val_loader):
            # Progress update after every 100 batches.
            if i % 100 == 0:
                print("--> batch {:} of {:}.".format(i, len(val_loader)))
            # Push the batch to gpu
            batch = [r.to(device) for r in batch]
            input_ids, attention_mask, labels = batch
            with torch.no_grad():
                # forward pass  
                preds = model(input_ids, attention_mask)
                preds, labels = preds.type(torch.FloatTensor), labels.type(torch.FloatTensor)
                # compute the loss between actual and predicted values
                loss = loss_func(preds, labels)
                # Check task 
                if task == "binary_cls" or task == "multi_cls":
                    preds = torch.round(preds)
                # Calculate all predictions and labels
                all_preds = torch.cat((all_preds, preds), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)
                # add on to the total loss
                running_loss += loss.item()
        # Calculate total running loss
        running_loss /= len(val_loader)
        # Calculate metrics
        if task == "binary_cls":
            metric_val = f1_score(y_true=all_labels, y_pred=all_preds, average="weighted", labels=np.arange(2))
        elif task == "multi_cls":
            metric_val = f1_score(y_true=all_labels, y_pred=all_preds, average="weighted", labels=np.arange(23))
        elif task == "regression":
            metric_val = mean_absolute_error(all_labels, all_preds)
        # Calculate running loss
        print(f"----> Validation loss {running_loss}")
        print(f"----> Time taken {time.time()-start}")
        print()
        return running_loss, metric_val
    
    def get_loss_func(self, task):
        # Return loss function based on machine learning task
        return nn.BCELoss() if task == "binary_cls" or task == "multi_cls" else nn.L1Loss()
    
    def get_metric_sign(self, task):
        # Returns the correct bound for the metric update
        return -1e16 if task == "binary_cls" or task == "multi_cls" else 1e16
    
    def get_eval_bound(self, task, model_val, best_val):
        # Evaluate the correct direction of the bound
        return eval(f"{model_val} > {best_val}") if task == "binary_cls" or task == "multi_cls" else eval(f"{model_val} < {best_val}")

    def run_opt(self):
        # Store best hyperparameters
        task_opt_hp = {}
        # Loop optimisation through different tasks
        for task in self.args.tasks:
            # Remove warning
            transformers.logging.set_verbosity_error()
            # Load model
            model = AutoModel.from_pretrained(args.pretrained_model, return_dict=False)
            # Adapt model
            model = get_model(task, model)
            # Move model to gpu
            model.to(device)
            # Apply tokenisation
            tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model)
             # Obtain text and labels
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data(self.args.folder, task, 
                                                                                                  self.args.anon)
            # Initialise Grid Search
            lr_list = [4e-5, 3e-5, 2e-5, 1e-5]
            batch_size_list = [16, 8, 4]
            best_lr, best_batch_size, best_metric_val = None, None, self.get_metric_sign(task=task)
            for _, lr in enumerate(lr_list):
                print(f"Grid Search begins - LR: {lr}")
                for _, batch_size in enumerate(batch_size_list):
                    # Generate tokens
                    train_tokens = generate_tokens(tokenizer, train_texts, self.args.max_seq_length)
                    val_tokens = generate_tokens(tokenizer, val_texts, self.args.max_seq_length)
                    # Load data
                    train_loader = create_dataloader(train_tokens, train_labels, batch_size, type="train")
                    val_loader = create_dataloader(val_tokens, val_labels, batch_size, type="val")
                    # Return best metrics against F1
                    metric_values = []
                    # Store losses
                    train_losses = []
                    val_losses = []
                    # Run GridSearchOpt
                    for epoch in range(self.args.num_epochs):
                        print('epoch {:} / {:}'.format(epoch + 1, self.args.num_epochs))
                        # Train model
                        train_loss = self.train(model, train_loader, lr, task)
                        # Evaluate model
                        val_loss, metric_val = self.validate(model, val_loader, task)
                        # Append training, validation losses and metric values for each epoch
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        metric_values.append(metric_val)
                        # Check for early stopping
                        if epoch >= 1:
                            if val_losses[epoch] > val_losses[epoch-1]:
                                break
                    # Save information for this task
                    print(f'[{task}] Metric values: {str(metric_values)}')
                    print(f'[{task}] training losses: {str(train_losses)}')
                    print(f'[{task}] validation losses: {str(val_losses)}')
                    print()
                    final_model_val = np.mean(metric_values)
                    # Add to log
                    self.log_results(task, lr, batch_size, train_losses, val_losses, metric_values)
                    # Check optimal metrics depending on task
                    if self.get_eval_bound(task=task, model_val=final_model_val, best_val=best_metric_val):
                        print("Metric value has improved - saving new model!")
                        best_lr = lr
                        best_batch_size = batch_size
                        torch.save(model.state_dict(), f'models/{str(task)}/{str(self.args.model_name)}.pt')
                        best_metric_val = final_model_val
                print(f"Grid Search finished for current LR. Next LR begins...")
            # Add best results to dictionary
            task_opt_hp[task] = {"Task": task, "Epoch": self.args.num_epochs, "Learning Rate": best_lr, "Batch Size": best_batch_size}
            self.logger.info(f"Best Hyperparameters: {task_opt_hp[task]}")
        return task_opt_hp


"""Run optimisation script."""

if __name__ == "__main__":
    print("=========================")
    print("Script starts...")
    print("=========================")
    print("Initialising model...")
    # Parse informatiom
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="echr")
    parser.add_argument("--anon", type=bool, default=False)
    parser.add_argument("--tasks", type=tuple, default = ["binary_cls", "multi_cls", "regression"], help = "The tasks to be worked on")
    parser.add_argument("--model_name", type=str, default = "echr_truncation_opt", help = "The path to save the model")
    parser.add_argument("--num_epochs", type=int, default = 5, help = "Number of epochs")
    parser.add_argument("--max_seq_length", type=int, default = 512, help = "The maximum length of the input sequence")
    parser.add_argument("--pretrained_model", type=str, default = "nlpaueb/legal-bert-small-uncased", help = "The path to the pretrained model")
    args = parser.parse_args()
    # Save information for this task
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    log_dir = "experiments/logs/opt"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, args.model_name + ".log"))
    fh.setLevel(logging.INFO)
    # Create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(fh)
    logger.info('folder: %s', args.folder)
    logger.info('anon: %s', args.anon)
    logger.info('model name: %s', args.model_name)
    logger.info('max sequence length: %d', args.max_seq_length)
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
    print("=========================")
    print("Execute Grid Search...")
    # Create Grid Search Model
    gs_opt_model = GridSearchOpt(args=args, logger=logger)
    # Run Grid Search
    optimal_hyp_tasks = gs_opt_model.run_opt()
    logger.info(f'Best Hyperparameters: {optimal_hyp_tasks}')
    print("Script finished!")
    print("=========================")