####################################################
# IMPORTING PACKAGES
####################################################

import jsonlines
import pandas as pd

####################################################
# DOWNLOADING DATA
####################################################

def jsonl_to_df(path):
    """
    Convert from jsonl to a dataframe
    """

    # Convert data from .jsonl to dataframe
    with jsonlines.open(path, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    df = pd.json_normalize(lst)

    # Function to fix list of dicts
    def fix_list_of_dicts(column, df):
        temp_df = df[column].apply(pd.Series)
        for i in range(len(temp_df.columns)):
            df_to_concat = pd.json_normalize(temp_df[i])
            df_to_concat.columns = [column + "." + str(col) + str(i+1) for col in df_to_concat.columns]
            df = df.join(df_to_concat)

        df = df.drop([column], axis=1)

        return df

    # Fixing list of dicts
    df = fix_list_of_dicts('casebody.data.opinions', df)
    df = fix_list_of_dicts('citations', df)

    return df

if __name__ == '__main__':
    new_mexico = jsonl_to_df(r'data\new_mexico.jsonl')
