# # Import

# +
import csv
import os
import sys

import numpy as np
import pandas as pd
from glob import glob
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
vectorize_type = sys.argv[2]
transformer_model = sys.argv[3]

max_vector_model_nums = config["vectorize"][vectorize_type][transformer_model]["max_model_num"]
vector_dims = config["vectorize"][vectorize_type][transformer_model]["dims"]

depression_type = "umap"

# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

if vectorize_type == "doc2vec":
    vector_object = f"Vectorize/{data_type}/{vectorize_type}/vector/"
elif vectorize_type == "sentenceBERT":
    vector_object = f"Vectorize/{data_type}/{vectorize_type}/{transformer_model}/vector/"
else:
    raise NotImplementedError

s3.download(vector_object)


# # Functions

def centralize_array(array):
    return array - np.mean(array, axis=0)


def normarize_array(array):
    return array / np.sqrt(np.sum(array * array, axis=1).reshape(-1, 1))


def normarize_vector(vector):
    return vector / np.sqrt(np.sum(vector * vector))


def get_average_vector(vectors):
    sum_vector = np.sum(vectors, axis=0)
    return normarize_vector(sum_vector)


# # Centralize Normalize

if vectorize_type == "doc2vec":
    vectors_path = f"{root_path_temporary}Vectorize/{data_type}/{vectorize_type}/vector"
    converted_vectors_path = (
        f"{root_path_temporary}Clustering/{data_type}/{vectorize_type}/vector"
    )
elif vectorize_type == "sentenceBERT":
    vectors_path = f"{root_path_temporary}Vectorize/{data_type}/{vectorize_type}/{transformer_model}/vector/{depression_type}"
    converted_vectors_path = f"{root_path_temporary}Clustering/{data_type}/{vectorize_type}/{transformer_model}/vector/{depression_type}"
else:
    raise NotImplementedError

for vector_model_num in range(max_vector_model_nums):
    for vector_dim in tqdm(vector_dims):
        vector = np.load(
            f"{vectors_path}/{vector_dim}/{vector_model_num}.npy",
        )
        centralized_vector = centralize_array(vector)
        normarized_vector = normarize_array(centralized_vector)

        np.save(
            make_filepath(
                f"{converted_vectors_path}/{vector_dim}/centralized/{vector_model_num}.npy"
            ),
            centralized_vector,
        )
        np.save(
            make_filepath(
                f"{converted_vectors_path}/{vector_dim}/normalized/{vector_model_num}.npy"
            ),
            normarized_vector,
        )

# # Upload files

s3.upload(
     converted_vectors_path,
)

s3.delete_local_all()


