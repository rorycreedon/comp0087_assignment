
"""
import packages
"""

from config import device
from sklearn.metrics import classification_report, mean_absolute_error
import torch
import logging
from model import get_model
from transformers import AutoTokenizer, AutoModel
from data_loader import load_data, generate_tokens, create_dataloader
from config import device
import argparse
import os
import transformers
from get_loss import get_loss_func
import time

"""
functions
"""

def test(test_loader, model, task, model_name):
    start = time.time()
    # load trained model
    model.load_state_dict(torch.load(f'models/{str(task)}/{str(model_name)}.pt'))

    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    running_loss = 0

    loss_func = get_loss_func(task)

    model.eval()

    # iterate over batches
    for i, batch in enumerate(test_loader):
        # progress update after every 100 batches.
        if i % 100 == 0:
            print("--> batch {:} of {:}.".format(i, len(test_loader)))

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

            if task == "binary_cls" or task == "multi_cls":
                preds = torch.round(preds)
    
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    running_loss = running_loss / len(test_loader)
    print(f"----> test loss {running_loss}")
    print(f"----> time taken {time.time()-start}")
    print()

    return all_preds, all_labels, running_loss

if __name__ == "__main__":
    print("=========================")
    print("script starts...")
    print()

    # parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default="echr")
    parser.add_argument("--tasks", type=list, default = ["binary_cls", "regression"], help = "The tasks to be worked on")
    parser.add_argument("--model_name", type=str, default = "echr_1", help = "The path to the saved model")
    parser.add_argument("--max_seq_length", type=int, default = 512, help = "The maximum length of the input sequence")
    parser.add_argument("--batch_size",type = int, default = 4, help = "Batch size")
    parser.add_argument("--pretrained_model", type=str, default = "nlpaueb/legal-bert-small-uncased", help = "The path to the pretrained model")

    args = parser.parse_args()

    log_dir = "experiments/logs/test"
    os.makedirs(log_dir, exist_ok=True)
    # check if the file exists in the logs folder
    model_path = os.path.join(log_dir, args.model_name + ".log")
    if os.path.isfile(model_path):
        raise ValueError("model_name has been used")

    # create a logger object
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    log_dir = "experiments/logs/test"
    fh = logging.FileHandler(os.path.join(log_dir, args.model_name + ".log"))
    fh.setLevel(logging.INFO)
    # create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the file handler to the logger
    logger.addHandler(fh)

    # save information for this task
    logger.info('folder: %s', args.folder)
    logger.info('model name: %s', args.model_name)
    logger.info('max sequence length: %d', args.max_seq_length)
    logger.info('batch size: %d', args.batch_size)
    logger.info('pretrained model: %s', args.pretrained_model)

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

        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data(args.folder, task)

        # remove warning
        transformers.logging.set_verbosity_error()

        # load model
        model = AutoModel.from_pretrained(args.pretrained_model, return_dict=False)
        # adapt model
        model = get_model(task, model)

        # move model to gpu
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

        test_tokens = generate_tokens(tokenizer, test_texts, args.max_seq_length)

        test_loader = create_dataloader(test_tokens, test_labels, args.batch_size, type="test")

        test_preds, test_labels, test_loss = test(test_loader, model, task, args.model_name)
        
        logger.info('[%s] test loss:%f', task, test_loss)

        if task == "binary_cls" or task == "multi_cls":
            report = classification_report(test_labels, test_preds)
            logger.info('[%s] classification report: \n%s', task, report)
        elif task == "regression":
            mae = mean_absolute_error(test_labels, test_preds)
            logger.info('[%s] mean absolute error: %f', task, mae)

    print("script finishes")
    print("=========================")