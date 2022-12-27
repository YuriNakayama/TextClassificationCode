# # Import

# +
import csv
import os
import sys

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
# -

# ## Add configuration file

sys.path.append("/home/jovyan/core/util/")

from util import * 


# # Function

def line_plot_error_1(
    data,
    error_low,
    error_upper,
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

    for column in data.columns:
        ax.errorbar(
            x=data.index.to_numpy(),
            y=data.loc[:, column].to_numpy(),
            yerr=np.stack(
                [
                    error_low.loc[:, column].to_numpy(),
                    error_upper.loc[:, column].to_numpy(),
                ],
            ),
            label=column,
            **layout["plot"]
        )

    if title is not None:
        ax.set_title(title, **layout["title"])

    if xlabel is not None:
        ax.set_xlabel(xlabel, **layout["label"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, **layout["label"])
    if xticks is not None:
        ax.set_xticks(ticks=xticks, **layout["ticks"])
    if yticks is not None:
        ax.set_yticks(ticks=yticks, **layout["ticks"])

    ax.legend(**layout["legend"])
    if path is not None:
        fig.savefig(make_filepath(path))


