# # Import

# +
import csv
import os
import sys

import numpy as np
import pandas as pd
from nltk import word_tokenize, download
from stop_words import get_stop_words
from nltk.corpus import stopwords
from tqdm import tqdm
# -

# ## Add configuration file

sys.path.append("/home/jovyan/core/config/")
sys.path.append("/home/jovyan/core/util/")

from ALL import config 
from util import *

# ## Set condition

tqdm.pandas()
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 50)

s3 = S3Manager()

data_type=sys.argv[1]

# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

# # Word tokenize

df["words"] = df.text.progress_apply(word_tokenize)

# # Remove long text

df["words_length"] = df.words.progress_apply(lambda x: len(x))

index = df[df["words_length"] < df["words_length"].quantile(0.996)].index

df = df.loc[index]

# # Remove stopwords

stop_words_add = ["would", "could", "should"]
stop_char = ["==", "--", "\'s", "''", "n't", "``","..", "...", "....", "'m", "'ve","'re", "'d", "'ll", "", "-+", "+-", "_/", "||", "__", "/|", "//"]
stop_words = set(stopwords.words("english") + get_stop_words("english") + stop_words_add + stop_char)

#     一文字以下の単語とstop_word, stop_charを削除
df["words_nonstop"] = df.words.progress_apply(
    lambda words: [
        word for word in words if word.lower() not in stop_words if len(word)> 1
    ]
)

df.words = df.words.progress_apply(lambda words: " ".join(words))
df.words_nonstop = df.words_nonstop.progress_apply(
    lambda words: " ".join(words)
)

# # output

# ## make file

with open(make_filepath(f"../../temporary/Preprocessing/{data_type}/class.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(class_labels)
df.to_csv(make_filepath(f"../../temporary/DataShaping/{data_type}/master.csv"))

# ## upload file

s3.upload(f"../../temporary/Preprocessing/{data_type}", f"DataShaping/{data_type}/")

s3.delete_local_all()


