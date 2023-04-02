# import required libraries
import torch
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

# Function for summarisation
def summarise(texts, tokenizer, model, device, min_summary_legnth, max_summary_length):
    """
    Summarise a batch of texts.

    Args:
        texts (list): list of texts to be summarised.
        tokenizer (transformers.models.led.tokenization_led_fast.LEDTokenizerFast): tokenizer for the summarisation model.
        model (transformers.models.led.modeling_led.LEDForConditionalGeneration): model for summarisation.
        device (str): device to run the model on. Either 'cpu' or 'cuda'.
        min_summary_length (int): minimum length of the summary.
        max_summary_length (int): maximum length of the summary.

    Returns:
        batch_summaries (list): batch of summaries.
    """
    # tokenize the batch of texts
    inputs = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True, max_length=10000).to(device)
    
    # generate summaries for the batch of texts
    summary_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_summary_length, min_length = min_summary_legnth)
    
    # decode the summary ids back into text
    batch_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return batch_summaries


# move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# longT5 model and tokenizer
model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")

# make folder if necessary
Path("data/echr/long_t5_summary").mkdir(parents=True, exist_ok=True)

# loop through all text in data
for set in ["train", "valid", "test"]:
    for name in ["non-anon"]:

        print(f"{name} {set} set")
        df = pd.read_pickle(f"data/echr/{name}_{set}.pkl")
        if "summary" not in df.columns:
            df["summary"] = ""
        i=0
        batch_size = 2
        num_batches = (len(df)//batch_size)+1
        
        # loop through summarise
        for j in tqdm(range(0, num_batches)):
            start_idx = j*batch_size
            end_idx = (j+1)*batch_size
            texts = df['text'][start_idx:end_idx].tolist()
            batch_summaries = summarise(texts, tokenizer, model, device, min(256, min(df['text'][start_idx:end_idx].str.split().apply(len))), 512)

            for k, summary in enumerate(batch_summaries):
                df["summary"][start_idx+k] = summary

            # save every 10 batches summaries in case of crashes
            df.to_pickle(f"data/echr/long_t5_summary/{name}_{set}.pkl")
            
        # save summary
        df.to_pickle(f"data/echr/long_t5_summary/{name}_{set}.pkl")