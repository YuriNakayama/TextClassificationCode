# # Import

# +
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pyLDAvis
from biterm.btm import oBTM
from biterm.utility import topic_summuary, vec_to_biterms  # helper functions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm
# -

# ## Add configuration file

sys.path.append("/home/jovyan/core/config/")
sys.path.append("/home/jovyan/core/util/")
sys.path.append("../PlotFunction/lineplot/")
sys.path.append("../PlotFunction/config/")

from ALL import config

# ## Set condition

tqdm.pandas()
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 50)

data_type = "AgNews"
from util import *

# # Read data

# +
df = pd.read_csv(f"../Preprocessing/data/{data_type}/master.csv", index_col=0)

# df = df.sample(frac=0.1)

class_num = config["data"][data_type]["class_num"]
texts = df.words_nonstop.tolist()
# vectorize texts
vec = CountVectorizer(stop_words="english", max_df=0.5, min_df=200)
X = vec.fit_transform(texts).toarray()

# get vocabulary
vocab = np.array(vec.get_feature_names())

# get biterms
biterms = vec_to_biterms(X)

# create btm
btm = oBTM(num_topics=class_num, V=vocab)

print("\n\n Train Online BTM ..")
for i in range(0, len(biterms), 1000):  # prozess chunk of 200 texts
    biterms_chunk = biterms[i : i + 1000]
    btm.fit(biterms_chunk, iterations=50)
topics = btm.transform(biterms)

path = f"data/{data_type}/biterm/prob.npy"
os.makedirs(os.path.dirname(path), exist_ok=True)
np.save(path, topics)
# -
pred = topics.argmax(axis=1)


adjusted_mutual_info_score(pred, df["class"].to_numpy())

send_line_notify(f"biterm")












