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
from util import *

# ## Set condition

tqdm.pandas()
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 50)

s3 = S3Manager()

data_type = "AgNews"

# # Read data

class_num = config["data"][data_type]["class_num"]

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path, index_col=0)

df = df.sample(frac=0.01)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

with open(labels_path, mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

# # Biterm

# ## Vectorize

# +
texts = df.words_nonstop.tolist()
# vectorize texts
vec = CountVectorizer(stop_words="english", max_df=0.5, min_df=0.03)
X = vec.fit_transform(texts).toarray()

# get vocabulary
vocab = np.array(vec.get_feature_names_out())
# -

# ## Clustering

# +
# get biterms
biterms = vec_to_biterms(X)

# create btm
btm = oBTM(num_topics=class_num, V=vocab)

print("\n\n Train Online BTM ..")
for i in range(0, len(biterms), 1000):  # prozess chunk of 200 texts
    biterms_chunk = biterms[i : i + 1000]
    btm.fit(biterms_chunk, iterations=50)
topics = btm.transform(biterms)
# -
# # Save

# ## Output 

prob_path = f"../temporary/Clustering/{data_type}/biterm/prob.npy"
np.save(make_filepath(prob_path), topics)

pred = topics.argmax(axis=1)
pred_path = f"../temporary/Clustering/{data_type}/biterm/pred.npy"
np.save(make_filepath(pred_path), pred_path)

# ## Upload

s3.upload(
    f"../temporary/Clustering/{data_type}/biterm/", f"Clustering/{data_type}/biterm/"
)

s3.delete_local_all()

send_line_notify(f"end {data_type} Biterm")





adjusted_mutual_info_score(pred, df["class"].to_numpy())








