# import required libraries
import torch
from transformers import AutoTokenizer, LEDForConditionalGeneration
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

# function for summarisation
def summarise(texts, tokenizer, model, device, min_summary_length, max_summary_length, gam=False):
    """
    Summarise a batch of texts.

    Args:
        texts (list): list of texts to be summarised.
        tokenizer (transformers.models.led.tokenization_led_fast.LEDTokenizerFast): tokenizer for the summarisation model.
        model (transformers.models.led.modeling_led.LEDForConditionalGeneration): model for summarisation.
        device (str): device to run the model on. Either 'cpu' or 'cuda'.
        min_summary_length (int): minimum length of the summary.
        max_summary_length (int): maximum length of the summary.
        gam (bool, optional): global attention mask. Defaults to False.

    Returns:
        batch_summaries (list): batch of summaries.
    """
    # tokenize the batch of texts
    inputs = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True, max_length = 10000).to(device)

    if (gam == True):
      # global attention on the first token (cf. Beltagy et al. 2020)
      global_attention_mask = torch.zeros_like(inputs['input_ids'])
      global_attention_mask[:, 0] = 1
      # generate summaries for the batch of texts with global attention on the first token
      summary_ids = model.generate(inputs['input_ids'], attention_mask=global_attention_mask, num_beams=3, max_length=max_summary_length, min_length = min_summary_length)
    else:
      # generate summaries for the batch of texts
      summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=max_summary_length, min_length = min_summary_length)

    # decode the summary ids back into text
    batch_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return batch_summaries

# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# led longformer model and tokenizer
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384").to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

# make folder if necessary
Path("data/echr/led_longformer").mkdir(parents=True, exist_ok=True)

# loop through all text in data
for set_ in ["train","valid", "test"]:
    for name in ["non-anon"]:
        print(f"{name} {set_} set")
        df = pd.read_pickle(f"data/echr/{name}_{set_}.pkl")
        if "summary" not in df.columns:
            df["summary"] = ""
        i=0
        batch_size = 2
        num_batches = (len(df)//batch_size)+1

        # Loop through summarise
        for j in tqdm(range(0,num_batches)):
            start_idx = j*batch_size
            end_idx = (j+1)*batch_size
            texts = df['text'][start_idx:end_idx].tolist()
            batch_summaries = summarise(texts, tokenizer, model, device, min(256, min(df['text'][start_idx:end_idx].str.split().apply(len))), 512)
            for k, summary in enumerate(batch_summaries):
                df["summary"][start_idx+k] = summary
            # save every 50 summaries in case of crashes
            if (j % 50==0) and (j>0):
                df.to_pickle(f"data/echr/led_longformer/{name}_{set_}.pkl")   
        # save summary
        df.to_pickle(f"data/echr/led_longformer/{name}_{set_}.pkl")
     

