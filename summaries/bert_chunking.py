# import required libraries
import pandas as pd
import torch
import transformers
from summarizer import Summarizer # bert extractive summarizer
from pathlib import Path

def chunk_text(text, max_length=512):
    """ 
    Split a large peice of text into chunks that are no longer than the max_length, but do not cut off sentences.

    Args:
        text (str): long document to be split into chunks.
        max_length (int, optional): max chunking length in words. Defaults to 512.

    Returns:
        chunks (list): list of chunked text.
    """
    # split the text into sentences
    sentences = text.split('. ')

    # initialize list to hold chunks and current chunk
    chunks = []
    current_chunk = ''

    # loop through each sentence and add it to the current chunk until the
    # length of the current chunk exceeds the max_length
    for sentence in sentences:
        words = sentence.split()
        if len(current_chunk.split()) + len(words) <= max_length:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '

    # append the final chunk to the list of chunks
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def chunked_extractive_summary(text, model, max_length=512):
    """
    Summarize a large peice of text using bert extractive summarizer via chunking.

    Args:
        text (str): long document to be summarized.
        model (summarizer.bert.Summarizer): bert extractive summarizer model.
        max_length (int, optional): max length of summarised text. Defaults to 512.

    Returns:
        summary (str): summary of the text.
    """
    
    # generate chunks
    chunks = chunk_text(text)

    # calculate the max output length for each chunk such that the 
    # total length of the summary is no longer than max_length
    max_output_length = 512 / len(chunks)

    # initialize list to hold summaries
    # loop through each chunk and generate a summary
    summaries = []
    for chunk in chunks:
        words_in_chunk = len(chunk.split())
        if words_in_chunk > max_output_length:
            ratio = max_output_length / words_in_chunk
            summaries.append(model(chunk, ratio=ratio))
        else:
            summaries.append(chunk)
    
    # join the summaries into a single string 
    summary = " ".join(summaries)

    return summary

# move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# bert summarizer model
model = Summarizer()

# make folder if necessary
Path("data/echr/chunking").mkdir(parents=True, exist_ok=True)

# loop through all long text in the echr datasets, generate summaries, add to dataframe and save
for set_ in ["train","valid", "test"]:
    for name in ["non-anon"]:
        print(f"{name} {set_} set")
        df = pd.read_pickle(f"data/echr/{name}_{set_}.pkl")
        if "summary" not in df.columns:
            df["summary"] = ""
        # loop through summarise
        for i in range(len(df["text"])):
            text = df["text"][i]
            summary = chunked_extractive_summary(text, model)
            while True:
                if len(summary.split()) <= 512:
                    break
                summary = chunked_extractive_summary(summary, model)
            df["summary"][i] = summary
            # save every 50 summaries in case of crashes
            if (i % 50==0) and (i>0):
                df.to_pickle(f"data/echr/chunking/{name}_{set_}.pkl")
        # save summary
        df.to_pickle(f"data/echr/chunking/{name}_{set_}.pkl")