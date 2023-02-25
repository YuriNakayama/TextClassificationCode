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


def _make_test(
    file_data: pd.DataFrame,
    file_num: int = 2,
    file_depth: int = 2,
    extension: str = "csv",
):
    _file_nums = [str(_file_num) for _file_num in range(file_num)]

    for _file_name in product(_file_nums, repeat=file_depth):
        _file_path = f"{root_path_temporary}/test/{'/'.join(_file_name)}/test.{extension}"
        file_data.to_csv(make_filepath(_file_path), index=False)


test_df = pd.DataFrame([[1, 2,3 ], [2, 3, 4], [5, 6, 7]])

_make_test(test_df, file_depth=2)


