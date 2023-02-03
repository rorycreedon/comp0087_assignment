import pandas as pd

#%%

def open_pickle(name):
    """
    Load pickle file into a dataframe
    """
    with open('data/' + name + '.pkl', 'rb') as file:
        df = pd.read_pickle(file)
        df.head

    return

open_pickle("new_mexico")

# %%
