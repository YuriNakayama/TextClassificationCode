# # Import

# +
import csv
import os
import pickle
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
# -

# ## Add configuration file

sys.path.append("/home/jovyan/core/config/")
sys.path.append("/home/jovyan/core/util/")
sys.path.append("/home/jovyan/Postprocessing/Function/")

from ALL import config 
from util import *
from extmath import row_norms

# ## Set condition

tqdm.pandas()
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 50)

s3 = S3Manager()

data_type = sys.argv[1]
vectorize_type =  sys.argv[2]
transformer_model = sys.argv[3]

# +
vector_dims = config["vectorize"][vectorize_type][transformer_model]["dims"]
normalizations = config["vectorize"][vectorize_type][transformer_model]["normalization"]
vector_model_nums = config["vectorize"][vectorize_type][transformer_model]["max_model_num"]

model_nums = config["clustering"]["gmm"]["max_model_num"]
covariance_types = config["clustering"]["gmm"]["covariance_types"]
topic_nums = config["data"][data_type_classifier(data_type)]["class_num"]
depression_type = "umap"
# -

# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

label = df["class"].to_numpy()

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

if vectorize_type == "doc2vec":
    vectors_path = f"Clustering/{data_type}/{vectorize_type}/vector"
    models_path = f"Clustering/{data_type}/{vectorize_type}/GMM/model/"
elif vectorize_type == "sentenceBERT":
    vectors_path = f"Clustering/{data_type}/{vectorize_type}/{transformer_model}/vector"
    models_path = f"Clustering/{data_type}/{vectorize_type}/{transformer_model}/GMM/model/"
else:
    raise NotImplementedError

s3.download(vectors_path)

s3.download(models_path)


# # functions

def _estimate_maharanobis_dist(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    means : array-like of shape (n_components, n_features)
    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    
    if covariance_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "tied":
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == "diag":
        precisions = precisions_chol**2
        log_prob = (
            np.sum((means**2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X**2, precisions.T)
        )

    elif covariance_type == "spherical":
        precisions = precisions_chol**2
        log_prob = (
            np.sum(means**2, 1) * precisions
            - 2 * np.dot(X, means.T * precisions)
            + np.outer(row_norms(X, squared=True), precisions)
        )
    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return log_prob


def gmm_value(gmm, vectors, label):
    pred = gmm.predict(vectors)
    prob = np.exp(gmm._estimate_weighted_log_prob(vectors))
    dist = _estimate_maharanobis_dist(
        vectors, gmm.means_, gmm.precisions_cholesky_, gmm.covariance_type
    )
    aic = gmm.aic(vectors)
    bic = gmm.bic(vectors)
    mi = adjusted_mutual_info_score(pred, label)
    logl = gmm.score(vectors, label)
    return {
        "pred": pred,
        "prob": prob,
        "dist": dist,
        "aic": aic,
        "bic": bic,
        "mi": mi,
        "logl": logl,
    }


# # Calculate Stats

if vectorize_type == "doc2vec":
    value_path = (
        f"/home/jovyan/temporary/Postprocessing/{data_type}/{vectorize_type}/GMM"
    )
if vectorize_type == "sentenceBERT":
    value_path = f"/home/jovyan/temporary/Postprocessing/{data_type}/{vectorize_type}/{transformer_model}/GMM"
else:
    raise NotImplementedError

for vector_model_num, vector_dim, normalization in tqdm(
    product(range(vector_model_nums), vector_dims, normalizations)
):
    vectors = np.load(
        f"{root_path_temporary}{vectors_path}/{depression_type}/{vector_dim}/{normalization}/{vector_model_num}.npy"
    )
    for covariance_type, topic_num in product(covariance_types, topic_nums):
        for model_num in range(model_nums):
            gmm = pickle.load(
                open(
                    f"{root_path_temporary}{models_path}{vector_dim}/{normalization}/{topic_num}/{covariance_type}/{model_num}.sav",
                    "rb",
                )
            )
            values = gmm_value(gmm, vectors, label)
            pred, prob, dist, *stat = values.items()
            # save
            for _name, _value in [pred, prob, dist]:
                save_path = f"{value_path}/{_name}/{vector_dim}/{normalization}/{vector_model_num}/{covariance_type}/{topic_num}/{model_num}.npy"
                np.save(make_filepath(save_path), _value)
            
            stat_path = f"{value_path}/stat/{vector_dim}/{normalization}/{vector_model_num}/{covariance_type}/{topic_num}/{model_num}.csv"
            with open(make_filepath(stat_path), "w") as f:
                writer = csv.DictWriter(f, dict(stat).keys())
                writer.writeheader()
                writer.writerow(dict(stat))
            

# ## upload file

s3.upload(value_path)

s3.delete_local_all()

send_line_notify(f"CalcStats.py {data_type} sentenceBERT {transformer_model}")


