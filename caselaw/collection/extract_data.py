import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)

    return df

def extract_rows(df):
    doc_ids = open("caselaw/collection/doc-ids.txt").read().splitlines()
    doc_ids = [int(id) for id in doc_ids]
    new_df = df[df["id"].isin(doc_ids)]
    if len(new_df) != 4029354:
        print("Case number not matched")
        print(set(doc_ids) ^ set(new_df["id"]))
        print("---")
        print(len(set(doc_ids) ^ set(new_df["id"])))
    print("---")
    print(len(new_df))
    # convert dataframe to json
    #new_df.to_json("data/cluster.json")

    return #new_df

df = load_data("/Users/tikantsoi/Downloads/opinion-clusters-2022-08-02.csv")
print(df.columns)
extract_rows(df)