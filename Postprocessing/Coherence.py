# # Import

# +
import csv
import os
import pickle
import sys

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
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

data_type=sys.argv[1]


# # Functions

class Corpus:
    def __init__(self, path=None, texts=None):
        self.path = path
        self.texts = texts
        self.dictionary = Dictionary(texts)

    def __iter__(self):
        if self.path is not None:
            for line in open(self.path):
                # assume there's one document per line, tokens separated by whitespace
                yield self.dictionary.doc2bow(line.lower().split())
        else:
            for line in self.texts:
                yield self.dictionary.doc2bow(line)


# # Read data

df_path = s3.download(f"Preprocessing/{data_type}/master.csv")

df = pd.read_csv(df_path[0], index_col=0)

label = df["class"].to_numpy()

# # Make Corpus

texts = df.words_nonstop.progress_apply(lambda x: x.split(" ") if x is not np.nan else [""]).tolist()
corpus = Corpus(texts=texts)
dictionary = Dictionary(texts)
dictionary.filter_extremes()

# # Calc Coherence

cm = CoherenceModel(
    topics=label.reshape(1, -1),
    corpus=corpus,
    dictionary=dictionary,
    texts=texts,
    coherence="c_v",
)
coherence_path = f"/home/jovyan/temporary/Preprocessing/{data_type}/coherence.csv"
pd.DataFrame([cm.get_coherence()]).to_csv(make_filepath(cohernce_path))


