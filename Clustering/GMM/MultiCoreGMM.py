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
    vectors_object = f"Clustering/{data_type}/{vectorize_type}/vector/{depression_type}/"
    models_object = f"Clustering/{data_type}/{vectorize_type}/GMM/model/"
    pred_object = f"Clustering/{data_type}/{vectorize_type}/GMM/pred/"
elif vectorize_type == "sentenceBERT":
    vectors_object = f"Clustering/{data_type}/{vectorize_type}/{transformer_model}/vector/{depression_type}/"
    models_object = f"Clustering/{data_type}/{vectorize_type}/{transformer_model}/GMM/model/"
    pred_object = f"Clustering/{data_type}/{vectorize_type}/{transformer_model}/GMM/pred/"
else:
    raise NotImplementedError


# # Clustering

def getGMM(vectors, n_components, covariance_type, seed, path):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=seed,
        max_iter=600,
        init_params="k-means++",
        reg_covar=1e-5,
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
        _vector_object = (
            f"{vectors_object}{vector_dim}/{normalization}/{vector_model_num}.npy"
        )
        _vectors_file_path = s3.download(_vector_object)
        vectors = np.load(_vectors_file_path[0])
        for covariance_type, n_component in product(covariance_types, n_components):
            iter_path = f"{vector_dim}/{normalization}/{n_component}/{covariance_type}/{model_num}"

            _model_file_path = f"{root_path_temporary}{models_object}{iter_path}.sav"
            pred = getGMM(
                vectors,
                seed=model_num,
                n_components=n_component,
                covariance_type=covariance_type,
                path=_model_file_path,
            )
            s3.upload(_model_file_path)
            s3.delete_local(_model_file_path)

            # save prediction
            _pred_file_path = f"{root_path_temporary}{pred_object}{iter_path}.npy"
            np.save(
                make_filepath(_pred_file_path),
                pred,
            )
            s3.upload(_pred_file_path)
            s3.delete_local(_pred_file_path)
        s3.delete_local(_vector_object)


# +
# with ProcessPoolExecutor(num_workers=os.cpu_count()) as executor:
#     executor.map(runGetGMM, model_nums)
# -

process_map(runGetGMM, model_nums, max_workers=os.cpu_count(), chunksize=1)

# # Upload files

s3.delete_local_all()

send_line_notify(f"end MultiCoreGMM.py {data_type} {vectorize_type} {transformer_model}")









