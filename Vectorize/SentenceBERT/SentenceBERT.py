# # Import

# +
import csv
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
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

# +
# sampling for test
# df.sample(n=1000, random_state=0)
# -

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

max_model_num = config["vectorize"]["sentenceBERT"][transformer_model]["max_model_num"]


# # Embedding

def get_sentenceBERT(texts,seed, path):
    model = SentenceTransformer(transformer_model)
    model.save(path)
    vectors = model.encode(texts)
    return vectors


# +
vectors_path = f"/home/jovyan/temporary/Vectorize/{data_type}/sentenceBERT/{transformer_model}/vector"
models_path = f"/home/jovyan/temporary/Vectorize/{data_type}/sentenceBERT/{transformer_model}/model"

for model_num in tqdm(range(max_model_num)):
    vectors = get_sentenceBERT(
        df.text.tolist(),
        seed=model_num,
        path=make_filepath(f"{models_path}/{model_num}"),
    )

    np.save(
        make_filepath(f"{vectors_path}/raw/{model_num}.npy"),
        np.stack(vectors),
    )
# -

# ## upload file

s3.upload(
    f"/home/jovyan/temporary/Vectorize/{data_type}/sentenceBERT/{transformer_model}/", 
    f"Vectorize/{data_type}/sentenceBERT/{transformer_model}/"
)

s3.delete_local_all()

send_line_notify(f"end SBERT {data_type} {transformer_model}")


