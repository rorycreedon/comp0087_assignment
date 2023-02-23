#%% Imports
import pandas as pd
import json
import glob

# %%
def open_json(path):
    """
    Load and concatenate json files in a directory
    """
    # create an empty list to store the DataFrames
    df_list = []

    for file_path in glob.glob("data/"+path+"/*.json"):
        with open(file_path, 'r') as f:
            # load the json data from the file object using the json module
            json_obj = json.load(f)
        # convert the json object to a pandas DataFrame
        df = pd.json_normalize(json_obj)
        # append the DataFrame to the list
        df_list.append(df)

    # concatenate all of the DataFrames in the list
    result_df = pd.concat(df_list, axis=0, ignore_index=True)

    return result_df

# %%
df = open_json("ECHR_Dataset/EN_train")

# %%
