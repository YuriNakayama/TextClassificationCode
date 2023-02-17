# # Import

# +
import csv
import os
import random
import sys
import pytest

import numpy as np
import pandas as pd
from tqdm import tqdm
# -

# ## Add configuration file

sys.path.append("/home/jovyan/core/config/")
sys.path.append("/home/jovyan/core/util/")

from ALL import config 
from util import *


# # Test Functions

def generate_1():
    names_keys = {"1": ["a", "b", "c"], "2": [1, 2, 3]}
    correct_dict = {
        "a": {1: {}, 2: {}, 3: {}},
        "b": {1: {}, 2: {}, 3: {}},
        "c": {1: {}, 2: {}, 3: {}},
    }
    md = multilayer_dict(names_keys)
    assert md.dict != correct_dict
