# # Import

# +
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
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

data_type="AgNewsTitle"
vectorize_type = "doc2vec"

# # Read data

df = pd.read_csv(
    f"../../Preprocessing/data/{data_type}/master.csv", index_col=0
)

with open(f"../../Preprocessing/data/{data_type}/class.csv", mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

max_vector_model_num = config["vectorize"][vectorize_type]["max_model_num"]
vector_dims = config["vectorize"][vectorize_type]["dims"]
normalization = config["vectorize"][vectorize_type]["normalization"]
model_nums = config["clustering"]["gmm"]["max_model_num"]
covariance_types = config["clustering"]["gmm"]["covariance_types"]

n_components = config["data"][data_type]["class_num"]


# # Clustering

def getGMM(vectors, n_components, covariance_type, seed, path):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        max_iter=200,
    )
    gmm.fit(vectors)
    # save model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(gmm, open(path, "wb"))
    pred = gmm.predict(vectors)
    return pred


vectors_path = f"../data/{data_type}/{vectorize_type}/vector"
models_path = f"../data/{data_type}/{vectorize_type}/GMM/model"
pred_path = f"../data/{data_type}/{vectorize_type}/GMM/pred"
for vector_model_num in range(max_vector_model_num):
    for vector_dim in tqdm(vector_dims):
        for model_num in range(model_nums):
            for covariance_type in covariance_types:
                vectors = np.load(f"{vectors_path}/{vector_dim}/{normalization}/{vector_model_num}.npy")

                pred = getGMM(
                    vectors,
                    seed=model_num,
                    n_components=n_components,
                    covariance_type=covariance_type,
                    path=f"{models_path}/{vector_dim}/{normalization}/{covariance_type}/{model_num}.sav",
                )

                # save prediction
                np.save(
                    make_filepath(
                        f"{pred_path}/{vector_dim}/{normalization}/{covariance_type}/{model_num}.npy"
                    ),
                    pred,
                )

send_line_notify(f"end {data_type} {vectorize_type}")


