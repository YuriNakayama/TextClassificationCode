# # Import

# +
import logging
import os
import shutil
from typing import List

import boto3
import requests
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# -

# # Load envs

load_dotenv()

s3_bucket_name = "text-classification-nakayama-bucket"
root_path = "/home/jovyan"
root_path_temporary = "/home/jovyan/temporary/"

# # Function


# ## file function

def make_filepath(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ## statistics function

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


# ## notification

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = os.getenv("LINE_NOTIFY_TOKEN")
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"{os.path.basename(os.getcwd())}: {notification_message}"}
    requests.post(line_notify_api, headers=headers, data=data)


# ## S3

class S3Manager:
    def __init__(self):
        self.files = {data_type: {} for data_type in ["upload", "download"]}

    def upload(
        self, file_path, object_name=None, bucket=s3_bucket_name, save_file=False
    ):
        """Upload a file to an S3 bucket

        :param file_path: File to upload
        :param object_name: S3 object name. If not specified then file_path is used
        :param bucket: Bucket to upload to. If not specified then default s3_bucket_name is used
        :param save_file: determine save file or not. If not specified then the uploaded file will not be saved.
        :return: file_name if file was uploaded, else False
        """

        def _upload_dir(_dir_path, _object_name, _bucket):
            for _root, _, _files in os.walk(_dir_path, topdown=False):
                if _files:
                    _root_path = _root.replace(_dir_path, _object_name)
                    _file_paths = [os.path.join(_root, _file) for _file in _files]
                    _objects = [os.path.join(_root_path, _file) for _file in _files]
                    for _object, _file_path in zip(_objects, _file_paths):
                        _s3_client.upload_file(_file_path, bucket, _object)

        # If S3 object_name was not specified, use subdirectry of "temporary" folder
        if object_name is None:
            object_name = os.path.abspath(file_path).replace(root_path_temporary, "")

        # Upload the file
        _s3_client = boto3.client("s3")
        try:
            if os.path.isfile(file_path):
                _response = _s3_client.upload_file(file_path, bucket, object_name)
            else:
                _upload_dir(file_path, object_name, bucket)
        except ClientError as e:
            logging.error(e)
            return False
        self.files["upload"][file_path] = save_file
        return file_path

    def download(
        self,
        object_name,
        file_path=None,
        s3_bucket_name=s3_bucket_name,
        save_file=False,
    ):
        """Download a file to an S3 bucket

        :param bucket: Bucket to download to
        :param file_path: File to download
        :param object_name: S3 object name. If not specified then file_path is used
        :param save_file: determine save file or not. If not specified then the uploaded file will not be saved.
        :return: True if file was uploaded, else False
        """
        _s3 = boto3.client("s3")
        try:
            _objects = self.ls(s3_bucket_name, object_name, recursive=True)
            _file_path_return = []
            if file_path is None:
                for _object in _objects:
                    _file_path = make_filepath(f"{root_path}/temporary/{_object}")
                    _file_path_return.append(_file_path)
                    _s3.download_file(s3_bucket_name, _object, _file_path)
                    self.files["download"][_file_path] = save_file
            else:
                for _object in _objects:
                    _file_path = make_filepath(_object.replace(object_name, file_path))
                    _file_path_return.append(_file_path)
                    _s3.download_file(s3_bucket_name, _object, _file_path)
                    self.files["download"][_file_path] = save_file

        except ClientError as e:
            logging.error(e)
            return False
        return _file_path_return

    def ls(self, bucket: str, prefix: str, recursive: bool = False) -> List[str]:
        """S3上のファイルリスト取得

        Args:
            bucket (str): バケット名
            prefix (str): バケット以降のパス
            recursive (bool): 再帰的にパスを取得するかどうか

        """
        paths: List[str] = []
        paths = self.__get_all_keys(bucket, prefix, recursive=recursive)
        return paths

    def __get_all_keys(
        self,
        bucket: str,
        prefix: str,
        keys: List = None,
        marker: str = "",
        recursive: bool = False,
    ) -> List[str]:
        """指定した prefix のすべての key の配列を返す

        Args:
            bucket (str): バケット名
            prefix (str): バケット以降のパス
            keys (List): 全パス取得用に用いる
            marker (str): 全パス取得用に用いる
            recursive (bool): 再帰的にパスを取得するかどうか

        """
        s3 = boto3.client("s3")
        if recursive:
            response = s3.list_objects(Bucket=bucket, Prefix=prefix, Marker=marker)
        else:
            response = s3.list_objects(
                Bucket=bucket, Prefix=prefix, Marker=marker, Delimiter="/"
            )

        # keyがNoneのときは初期化
        if keys is None:
            keys = []

        if "CommonPrefixes" in response:
            # Delimiterが'/'のときはフォルダがKeyに含まれない
            keys.extend([content["Prefix"] for content in response["CommonPrefixes"]])
        if "Contents" in response:  # 該当する key がないと response に 'Contents' が含まれない
            keys.extend([content["Key"] for content in response["Contents"]])
            if "IsTruncated" in response:
                return self.__get_all_keys(
                    bucket=bucket,
                    prefix=prefix,
                    keys=keys,
                    marker=keys[-1],
                    recursive=recursive,
                )
        return keys

    def delete_local_all(self):
        def _delete_file_or_folder(_path):
            if os.path.exists(_path):
                if os.path.isfile(_path):
                    os.remove(_path)
                else:
                    shutil.rmtree(_path)
                return True
            else:
                return False

        for _type, _file_dict in self.files.items():
            if _file_dict:
                for _file, _save in _file_dict.items():
                    if not _save:
                        _delete_file_or_folder(_file)
                        print(_file)
