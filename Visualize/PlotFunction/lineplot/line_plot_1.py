# # Import

# +
import csv
import os
import sys

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as npa
import pandas as pd
import seaborn as sns
from tqdm import tqdm
# -

# ## Add configuration file

sys.path.append("/home/jovyan/core/util/")

from util import * 


# # Function

def line_plot_1(
    data,
    fig,
    ax,
    layout,
    path=None,
    title=None,
    xlabel=None,
    ylabel=None,
    xticks=None,
    yticks=None,
):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sns.lineplot(data, ax=ax, dashes=False)
    
    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
    if yticks is not None:
        ax.set_yticks(ticks=yticks)

    ax.legend(**layout["legend"])
    if path is not None:
        fig.savefig(make_filepath(path))


