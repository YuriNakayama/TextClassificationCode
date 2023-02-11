# # Import

# +
import csv
import os
import sys

import numpy as np
import pandas as pd
import umap
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
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

data_type = sys.argv[1]

transformer_model = sys.argv[2]

# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

s3.download(f"Vectorize/{data_type}/sentenceBERT/{transformer_model}")

max_model_num = config["vectorize"]["sentenceBERT"][transformer_model]["max_model_num"]
vector_dims = config["vectorize"]["sentenceBERT"][transformer_model]["dims"]

# # Dimension Depression

# +
vector_path = (
    f"/home/jovyan/temporary/Vectorize/{data_type}/sentenceBERT/{transformer_model}/vector"
)

for model_num in range(max_model_num):
    vectors = np.load(
        f"{vector_path}/raw/{model_num}.npy",
    )
    default_vector_dim = vectors.shape[1]
    for vector_dim in tqdm(vector_dims):
        if vector_dim < default_vector_dim:
            reduced_vectors = PCA(
                n_components=vector_dim, random_state=model_num
            ).fit_transform(vectors)

            np.save(
                make_filepath(f"{vector_path}/pca/{vector_dim}/{model_num}.npy"),
                reduced_vectors,
            )
# -

np.save(
    make_filepath(f"{vector_path}/pca/{vectors.shape[1]}/{model_num}.npy"), vectors
)

# +
vector_path = (
    f"/home/jovyan/temporary/Vectorize/{data_type}/sentenceBERT/{transformer_model}/vector"
)

for model_num in range(max_model_num):
    vectors = np.load(
        f"{vector_path}/raw/{model_num}.npy",
    )
    default_vector_dim = vectors.shape[1]
    for vector_dim in tqdm(vector_dims):
        if vector_dim < default_vector_dim:
            reduced_vectors = umap.UMAP(
                n_components=vector_dim, random_state=model_num
            ).fit_transform(vectors)
            np.save(
                make_filepath(f"{vector_path}/umap/{vector_dim}/{model_num}.npy"),
                reduced_vectors,
            )
# -

np.save(
    make_filepath(f"{vector_path}/umap/{vectors.shape[1]}/{model_num}.npy"), vectors
)

# ## upload file

s3.upload(
    f"/home/jovyan/temporary/Vectorize/{data_type}/sentenceBERT/{transformer_model}/vector", 
    f"Vectorize/{data_type}/sentenceBERT/{transformer_model}/vector"
)

s3.delete_local_all()

send_line_notify(f"end DimensionDepression.py {data_type} sentenceBERT {transformer_model}")






