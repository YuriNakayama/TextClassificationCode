# # Import

# +
import csv
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
from itertools import product

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

s3 = S3Manager()

data_type=sys.argv[1]
vectorize_type =sys.argv[2]
transformer_model = sys.argv[3]

max_vector_model_num = config["vectorize"][vectorize_type][transformer_model]["max_model_num"]
vector_dims = config["vectorize"][vectorize_type][transformer_model]["dims"]
normalizations = config["vectorize"][vectorize_type][transformer_model]["normalization"]
max_model_num = config["clustering"]["gmm"]["max_model_num"]
covariance_types = config["clustering"]["gmm"]["covariance_types"]

n_components = config["data"][data_type_classifier(data_type)]["class_num"]

depression_type = "umap"

# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

if vectorize_type == "doc2vec":
    vector_object = f"Clustering/{data_type}/{vectorize_type}/vector/{depression_type}/"
    vectors_path = f"/home/jovyan/temporary/Clustering{data_type}/{vectorize_type}/vector/{depression_type}/"
    models_path = f"/home/jovyan/temporary/Clustering/{data_type}/{vectorize_type}/GMM/model/"
    pred_path = f"/home/jovyan/temporary/Clustering/{data_type}/{vectorize_type}/GMM/pred/"
elif vectorize_type == "sentenceBERT":
    vector_object = f"Clustering/{data_type}/{vectorize_type}/{transformer_model}/vector/{depression_type}/"
    vectors_path = f"/home/jovyan/temporary/Clustering/{data_type}/{vectorize_type}/{transformer_model}/vector/{depression_type}/"
    models_path = f"/home/jovyan/temporary/Clustering/{data_type}/{vectorize_type}/{transformer_model}/GMM/model/"
    pred_path = f"/home/jovyan/temporary/Clustering/{data_type}/{vectorize_type}/{transformer_model}/GMM/pred/"
else:
    raise NotImplementedError

s3.download(vector_object)


# # Clustering

def getGMM(vectors, n_components, covariance_type, seed, path):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        max_iter=400,
        init_params="k-means++",
        n_init=1,
    )
    gmm.fit(vectors)
    # save model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(gmm, open(path, "wb"))
    pred = gmm.predict(vectors)
    return pred


model_nums = [model_num for model_num in range(max_model_num)]


def runGetGMM(model_num):
    for vector_model_num, vector_dim, normalization in product(
        range(max_vector_model_num), vector_dims, normalizations
    ):
        vectors = np.load(
            f"{vectors_path}{vector_dim}/{normalization}/{vector_model_num}.npy"
        )
        for covariance_type, n_component in product(covariance_types, n_components):
            pred = getGMM(
                vectors,
                seed=model_num,
                n_components=n_component,
                covariance_type=covariance_type,
                path=f"{models_path}{vector_dim}/{normalization}/{n_component}/{covariance_type}/{model_num}.sav",
            )

            # save prediction
            np.save(
                make_filepath(
                    f"{pred_path}{vector_dim}/{normalization}/{n_component}/{covariance_type}/{model_num}.npy"
                ),
                pred,
            )


# +
# with ProcessPoolExecutor(num_workers=os.cpu_count()) as executor:
#     executor.map(runGetGMM, model_nums)
# -

r = process_map(runGetGMM, model_nums, max_workers=os.cpu_count(), chunksize=100)

# # Upload files

s3.upload(
     models_path,
)

s3.upload(
     pred_path,
)

send_line_notify(f"end MultiGMM{data_type} {vectorize_type} {transformer_model}")









for vector_model_num in range(max_vector_model_num):
    for vector_dim in tqdm(vector_dims):
        for normalization in normalizations:
            vectors = np.load(
                f"{vectors_path}{vector_dim}/{normalization}/{vector_model_num}.npy"
            )
            
            for model_num in range(model_nums):
                for covariance_type in covariance_types:
                    for n_component in n_components:
                        pred = getGMM(
                            vectors,
                            seed=model_num,
                            n_components=n_components,
                            covariance_type=covariance_type,
                            path=f"{models_path}{vector_dim}/{normalization}/{n_component}/{covariance_type}/{model_num}.sav",
                        )

                        # save prediction
                        np.save(
                            make_filepath(
                                f"{pred_path}{vector_dim}/{normalization}/{n_component}/{covariance_type}/{model_num}.npy"
                            ),
                            pred,
                        )
