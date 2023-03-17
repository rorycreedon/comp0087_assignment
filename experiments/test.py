
"""
import packages
"""

from experiments.config import device
from sklearn.metrics import classification_report
import torch
import logging
from model import get_model
from transformers import AutoTokenizer, AutoModel
from data_loader import load_data, generate_tokens, create_dataloader
from config import device
import argparse
import os
import transformers

"""
functions
"""

def test(test_loader, model, task, model_name):
    # load trained model
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
    
            all_preds = torch.cat((all_preds, preds), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    return all_preds, all_labels

if __name__ == "__main__":
    print("=========================")
    print("script starts...")
    print()

    # parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default="echr")
    parser.add_argument("--tasks", type=tuple, default = ("binary_cls", "regression"), help = "The tasks to be worked on")
    parser.add_argument("--model_name", type=str, default = "legalbert_echr_truncation-1", help = "The path to save the model")
    parser.add_argument("--max_seq_length", type=int, default = 512, help = "The maximum length of the input sequence")
    parser.add_argument("--batch_size",type = int, default = 4, help = "Batch size")
    parser.add_argument("--pretrained_model", type=str, default = "nlpaueb/legal-bert-small-uncased", help = "The path to the pretrained model")

    args = parser.parse_args()

    # create a logger object
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    log_dir = "experiments/logs/test"
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
    logger.info('model name: %s', args.model_name)
    logger.info('max sequence length: %d', args.max_seq_length)
    logger.info('number of epochs: %d', args.num_epochs)
    logger.info('learning rate: %f', args.lr)
    logger.info('batch size: %d', args.batch_size)
    logger.info('pretrained model: %s', args.pretrained_model)

    # use gpu
    print(f"using {device}")
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
        model = get_model(model, task)
        # move model to gpu
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

        test_tokens = generate_tokens(tokenizer, test_texts, args.max_seq_length)

        test_loader = create_dataloader(test_tokens, test_labels, args.batch_size, type="test")

        test_preds, test_labels = test(test_loader, model, task, model_name="")

        if task=="binary_cls":
            report = classification_report(test_labels, test_preds)
            logger.info('[%s] Classification report:\n%s', task, report)

    print("script finishes")
    print("=========================")