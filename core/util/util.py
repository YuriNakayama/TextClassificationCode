# # Import

# +
import os
import sys
import copy

import requests
from dotenv import load_dotenv
# -

# # Load envs

load_dotenv()


# # Function

def make_filepath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_describe(df, axis=0):
    """
    pd.DataFrameの統計値を取得する。
    Parameters
    ----------
    df : pd.DataFrame
    axis : 0, 1
    Returns
    -------
    describe : dict(pd.Series)
        集約された統計値
    keys: list[str]
        統計値のリスト
    """
    describe = {
        "mean": df.mean(axis=axis),
        "median": df.median(axis=axis),
        "std": df.std(axis=axis),
        "var": df.var(axis=axis),
        "75": df.quantile(0.75, axis=axis),
        "25": df.quantile(0.25, axis=axis),
    }
    keys = describe.keys()
    return describe, keys


def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = os.getenv("LINE_NOTIFY_TOKEN")
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {
        "message": f"{os.path.basename(os.getcwd())}: {notification_message}"
    }
    requests.post(line_notify_api, headers=headers, data=data)


def make_multilayer_dict(keys: list):
    def _multilayer_dict_recursive(d: dict, keys: list):
        if not keys:
            return d, []
        else:
            return _multilayer_dict_recursive(
                {key: copy.deepcopy(d) for key in keys[-1]}, keys[:-1]
            )

    multilayer_dict, _ = _multilayer_dict_recursive(dict(), keys)
    return multilayer_dict


