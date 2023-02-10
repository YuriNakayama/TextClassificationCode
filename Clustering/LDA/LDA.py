# # Import

# +
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from smart_open import open
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

model_nums = config["clustering"]["LDA"]["max_model_num"]

n_components = config["data"][data_type_classifier(data_type)]["class_num"]

# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

labels_path = s3.download(f"Preprocessing/{data_type}/class.csv")

with open(labels_path[0], mode="r") as f:
    reader = csv.reader(f)
    class_labels = [label for label in reader]

s3.download(vectors_path)


# # LDA

class Corpus:
    def __init__(self, path=None, texts=None):
        self.path = path
        self.texts = texts
        self.dictionary = Dictionary(texts)
        self.dictionary.filter_extremes()

    def __iter__(self):
        if self.path is not None:
            for line in open(self.path):
                # assume there's one document per line, tokens separated by whitespace
                yield self.dictionary.doc2bow(line.lower().split())
        else:
            for line in self.texts:
                yield self.dictionary.doc2bow(line)
                
    def __len__(self):
        return len(self.texts)


texts = df.words_nonstop.progress_apply(lambda x: x.split(' ') if x is not np.nan else [""]).tolist()
corpus = Corpus(texts=texts)
dictionary = Dictionary(texts)
dictionary.filter_extremes()

os.makedirs(os.path.dirname(f"../../temporary/{data_type}/LDA/"), exist_ok=True)
pickle.dump(dictionary, open(f"../../temporary/{data_type}/LDA/dictionary.sav", "wb"))
pickle.dump(corpus, open(f"../../temporary/{data_type}/LDA/corpus.sav", "wb"))

label = df["class"].to_numpy()


def getLDA(corpus,dictionary, n_components, seed, path):
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_components,
        alpha="auto",
        eval_every=5,
        random_state=seed,
    )
    # save model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lda.save(path)
    pred = [lda[docBow] for docBow in corpus]
    return pred, lda


# +
models_path = f"../../temporary/Clustering/{data_type}/LDA/model/"
pred_path = f"../../temporary/Clustering/{data_type}/LDA/pred/"
prob_path = f"../../temporary/Clustering/{data_type}/LDA/prob/"

for model_num in tqdm(range(model_nums)):
    prob, lda = getLDA(
        corpus=corpus,
        dictionary=dictionary,
        n_components=n_conmponents,
        seed=model_num,
        path=f"{models_path}{model_num}"
    )
#     save prediction
    probDf = pd.DataFrame([dict(row) for row in prob]).fillna(0)
    
    os.makedirs(f"{prob_path}", exist_ok=True)
    probDf.to_csv(f"{prob_path}{model_num}.csv")
    
    pred = probDf.idxmax(axis=1).to_numpy()
    os.makedirs(f"{pred_path}", exist_ok=True)
    np.save(f"{pred_path}{model_num}.npy", pred)
# -

# ## upload file

s3.upload(
    f"../../temporary/Clustering/{data_type}/LDA/", 
)

s3.delete_local_all()

send_line_notify(f"end {data_type} LDA")


