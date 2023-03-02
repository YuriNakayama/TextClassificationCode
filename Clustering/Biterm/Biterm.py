# # Import

# +
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm

from biterm.btm import oBTM
from biterm.utility import topic_summuary, vec_to_biterms  # helper functions
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

sampling_nums = [1, 2, 4, 8, 16, 32, 64, 128]

# data_types = [f"20NewsSampled{sampling_num}" for sampling_num in sampling_nums]
data_types = ["TweetTopic", "TweetFinance"]
clustering_model = "biterm"

data_type = data_types[0]

model_nums = config["clustering"][clustering_model]["max_model_num"]
topic_nums = {
    data_type: config["data"][data_type_classifier(data_type)]["class_num"]
    for data_type in data_types
}

# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

label = df["class"].to_numpy()

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

base_pred_path = f"Clustering/{data_type}/BERTopic/pred"
base_model_path = f"Clustering/{data_type}BERTopic/model"
base_prob_path = f"Postprocessing/{data_type}/BERTopic/prob"

# # Biterm

# ## Vectorize

# +
texts = df.words_nonstop.tolist()
# vectorize texts
vec = CountVectorizer(stop_words="english")
X = vec.fit_transform(texts).toarray()

# get vocabulary
vocab = np.array(vec.get_feature_names_out())
# -

# ## Clustering

for model_num in range(model_nums):
    # get biterms
    biterms = vec_to_biterms(X)

    # create btm
    btm = oBTM(num_topics=topic_nums[data_type][0], V=vocab)

    print("\n\n Train Online BTM ..")
    for i in range(0, len(biterms), 1000):  # prozess chunk of 200 texts
        biterms_chunk = biterms[i : i + 1000]
        btm.fit(biterms_chunk, iterations=50)
    topics = btm.transform(biterms)
    
    # output
    prob_path = f"/home/jovyan/temporary/Clustering/{data_type}/biterm/prob/{model_num}.npy"
    np.save(make_filepath(prob_path), topics)
    pred = topics.argmax(axis=1)
    pred_path = f"/home/jovyan/temporary/Clustering/{data_type}/biterm/pred/{model_num}.npy"
    np.save(make_filepath(pred_path), pred_path)
# +
model_num = 0
# get biterms
biterms = vec_to_biterms(X)

# create btm
btm = oBTM(num_topics=topic_nums[data_type][0], V=vocab)

print("\n\n Train Online BTM ..")
for i in range(0, len(biterms), 1000):  # prozess chunk of 200 texts
    biterms_chunk = biterms[i : i + 1000]
    btm.fit(biterms_chunk, iterations=50)
topics = btm.transform(biterms)

# output
prob_path = f"/home/jovyan/temporary/Clustering/{data_type}/biterm/prob/{model_num}.npy"
np.save(make_filepath(prob_path), topics)
pred = topics.argmax(axis=1)
pred_path = f"/home/jovyan/temporary/Clustering/{data_type}/biterm/pred/{model_num}.npy"
np.save(make_filepath(pred_path), pred_path)
# -

# # Upload

s3.upload(
    f"/home/jovyan/temporary/Clustering/{data_type}/biterm"
)

s3.delete_local_all()

send_line_notify(f"end {data_type} Biterm")





adjusted_mutual_info_score(pred, df["class"].to_numpy())








