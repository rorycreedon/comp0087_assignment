####################################################
# IMPORTING PACKAGES
####################################################

import jsonlines
import pandas as pd
import zipfile
from io import BytesIO
import requests
import lzma
import os
import shutil
from pathlib import Path
from tqdm import tqdm

####################################################
# DOWNLOADING DATA
####################################################


def download_extract_data(url, name):
	"""
	Download data from case.law website and extract into a .jsonl file.

	Params:
	`url` (str): URL of zip file
	`name` (str): name that .jsonl file will have
	"""

	# Download and unzip files from website
	response = requests.get(url)
	with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
		zip_ref.extractall("temp/")

	# Get name of new folder made
	folder = next(os.walk("temp/"))[1][0]

	# Extract .jsonl file from .xz file
	with lzma.open("temp/" + folder + "/data/data.jsonl.xz") as f, open(
		"temp/" + name + ".jsonl", "wb"
	) as f_out:
		file_content = f.read()
		f_out.write(file_content)

	# Delete unextracted data
	shutil.rmtree("temp/" + folder)


def jsonl_to_df(name):
	"""
	Convert from .jsonl file to dataframe

	Params:
	`name` (str): name that saved .jsonl file has
	"""

	# Convert data from .jsonl to dataframe
	with jsonlines.open("temp/" + name + ".jsonl", "r") as jsonl_f:
		lst = [obj for obj in jsonl_f]
	df = pd.json_normalize(lst)

	# Function to fix list of dicts
	def fix_list_of_dicts(column, df):
		temp_df = df[column].apply(pd.Series)
		for i in range(len(temp_df.columns)):
			df_to_concat = pd.json_normalize(temp_df[i])
			df_to_concat.columns = [
				column + "." + str(col) + str(i + 1) for col in df_to_concat.columns
			]
			df = df.join(df_to_concat)

		df = df.drop([column], axis=1)

		return df

	# Fixing list of dicts
	df = fix_list_of_dicts("casebody.data.opinions", df)
	df = fix_list_of_dicts("citations", df)

	# Delete .jsonl file
	os.remove("temp/" + name + ".jsonl")

	df.to_pickle("data/" + name + ".pkl")


if __name__ == "__main__":

	# Make folders if necessary
	Path("data/").mkdir(parents=True, exist_ok=True)
	Path("temp/").mkdir(parents=True, exist_ok=True)

	# Links and relevant states to download
	urls = ["https://case.law/download/bulk_exports/latest/by_jurisdiction/case_text_open/nm/nm_text.zip", "https://case.law/download/bulk_exports/latest/by_jurisdiction/case_text_open/ark/ark_text.zip"]
	states = ["new_mexico", "arkansas"]

	# Downloading and saving data
	for i in tqdm(range(len(urls))):
		download_extract_data(urls[i], states[i])
		jsonl_to_df(states[i])