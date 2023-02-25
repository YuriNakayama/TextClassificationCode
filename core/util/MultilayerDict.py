# # Import

# +
import copy
import csv
import itertools
import os
from typing import Dict, List, Type

import numpy as np
from sympy.combinatorics import Permutation
# -

# # Function


# ## multilayer dict

class MultilayerDict:
    """
    MultilayerDict
    """

    names: List
    names_keys: Dict
    dictionary: Dict

    def __init__(
        self,
        names_keys: Dict | None = None,
        dictionary: Dict | None = None,
        dir_path: str | None = None,
        names: List | None = None,
        extension: str = "csv",
        basis_depth: str = "max",
    ):
        if names_keys is not None:
            self.names = list(names_keys.keys())
            self.names_keys = names_keys
            if dictionary is not None:
                self.dictionary = dictionary
            else:
                self.dictionary = self.make_multilayer_dict(list(names_keys.values()))
        elif dir_path is not None:
            self.dictionary, _keys, self.names = self.read_dirs(
                dir_path,
                names=names,
            )
        else:
            raise ValueError("Both names_keys and dir_path are None.")

    def __check_all_none(self, *args) -> bool:
        if all(_v is None for _v in args):
            return True
        else:
            return False

    def __check_one_not_none(self, *args) -> bool:
        if sum(1 for arg in args if arg is not None) == 1:
            return True
        else:
            return False

    def make_multilayer_dict(self, keys: List):
        def _multilayer_dict_recursive(_d: Dict, _keys: List):
            if not _keys:
                return _d, []
            else:
                return _multilayer_dict_recursive(
                    {_key: copy.deepcopy(_d) for _key in _keys[-1]}, _keys[:-1]
                )

        _multilayer_dict, _ = _multilayer_dict_recursive(dict(), keys)
        return _multilayer_dict

    def name_is_in(self, name) -> bool:
        return name in self.name

    def keys_exist(self, keys: List | Tuple) -> bool:
        def __is_accessible(_d: Dict, _t: List | Tuple) -> bool:
            if len(_t) == 1:
                return _t[0] in _d
            if _t[0] in _d:
                return __is_accessible(_d[_t[0]], _t[1:])
            else:
                return False

        return __is_accessible(self.dictionary, keys)

    def loc(self, key_list: List):
        def _loc_recursive(_val, _key_list: list):
            if not _key_list:
                return _val
            else:
                return _loc_recursive(_val[_key_list[0]], _key_list[1:])

        return _loc_recursive(self.dictionary, key_list)

    def update(self, key_list: List, val) -> None:
        _d = self.dictionary
        for key in key_list[:-1]:
            _d = _d[key]
        _d[key_list[-1]] = val

    def read_dirs(self, dir_path, names=None, extension="csv", basis_depth="max"):
        def _one_dimensional_csv_to_dict(
            file_path, encoding="utf8", newline="", delimiter=",", quotechar="|"
        ):
            print(file_path)
            with open(file_path, mode="r", encoding=encoding, newline=newline) as _f:
                _reader = csv.reader(_f, delimiter=delimiter, quotechar=quotechar)
                _dict = dict(_reader)
            return _dict

        def _one_dimensional_json_to_dict(
            file_path,
            encoding="utf8",
            newline="",
        ):
            with open(file_path, mode="r", encoding=encoding, newline=newline) as _f:
                _dict = json.load(_f)
            return _dict

        def _get_uniform_dirs(__path_dirs, _basis_depth):
            # 最も深いパスを基準にMultilayerDictを生成する
            if _basis_depth == "max":
                _depth = max(map(len, __path_dirs.values()))
                _uniform_dirs = {
                    _path: _dir
                    for _path, _dir in __path_dirs.items()
                    if len(_dir) == _depth
                }
                return _depth, _uniform_dirs
            elif _basis_depth == "min":
                _depth = min(map(len, __path_dirs.values()))
                _uniform_dirs = {
                    _path: _dir
                    for _path, _dir in __path_dirs.items()
                    if len(_dir) == _depth
                }
                return _depth, _uniform_dirs
            elif isinstance(_basis_depth, int) & (_basis_depth > 0):
                _depth = _basis_depth
                _uniform_dirs = {
                    _path: _dir
                    for _path, _dir in __path_dirs.items()
                    if len(_dir) == _depth
                }
                return _depth, _uniform_dirs
            else:
                raise NotImplementedError

        # 読み込みディレクトリの中のファイルパスを探索し,ファイルパスからdictのkeyを生成
        _path_dirs = {}
        _abs_dir_path = os.path.abspath(dir_path)
        for _root, _, _file_names in os.walk(_abs_dir_path, followlinks=True):
            if _file_names:
                for _file_name in _file_names:
                    _file_path = f"{_root}/{_file_name}"
                    _file_name = os.path.splitext(os.path.basename(_file_name))[0]
                    _path_dirs[_file_path] = (
                        _file_path.replace(_abs_dir_path, "")
                    ).split("/")[1:]

        _depth, _uniform_path_dirs = _get_uniform_dirs(_path_dirs, basis_depth)

        # ファイルの読み込み
        _file_keys = set()
        if extension == "csv":
            for _path, _dir_list in _uniform_path_dirs.items():
                print(_path)
                print(_dir_list)
                _file_dict = _one_dimensional_csv_to_dict(_path)
                md.update(_dir_list, _file_dict)
                _file_keys = _file_keys & set(_file_dict)
        elif extension == "json":
            for _path, _dir_list in _uniform_path_dirs.items():
                _file_dict = _one_dimensional_json_to_dict(_path)
                md.update(_dir_list, _file_dict)
                _file_keys = _file_keys & set(_file_dict)
        else:
            raise NotImplementedError

        _keys = [list(set(_)) for _ in zip(*_uniform_path_dirs.values())]
        _keys.append(_file_keys)

        if names:
            _names = range(_depth + 1)
        else:
            if len(names) == _depth + 1:
                _names = names
            else:
                raise ValueError(
                    f"length mismatch. basis_depth is {basis_depth}, but length of name is {len(names)}."
                )
        _md = self.make_multilayer_dict(_names)
        return _md, _keys, _names

    def drop_names(
        self, names: List | None = None, loc: int | None = None, inplace: bool = False
    ):
        if self.__check_all_none(names, loc):
            raise ValueError("At least one argument must not be None.")

        if not self.__check_one_not_none(names, loc):
            raise ValueError("Multiple variables are specified.")

        if names is not None:
            if self.names[-len(names) :] != names:
                raise ValueError(
                    f"The given names ({names}) do not match the variable ({self.names})."
                )
            if inplace:
                self.names = [_ for _ in self.names[: -len(names)]]
                self.names_keys = {
                    _name: _val
                    for _name, _val in self.names_keys.items()
                    if _name in self.names
                }
            else:
                _names = [_ for _ in self.names[: -len(names)]]
                _names_keys = {
                    _name: _val
                    for _name, _val in self.names_keys.items()
                    if _name in _names
                }
                _dictionary = copy.deepcopy(self.dictionary)
                return MultilayerDict(_names_keys, _dictionary)

        elif loc is not None:
            if len(self.names_keys) < loc:
                raise ValueError(
                    f"Value of loc ({loc}) exceed length of names ({len(self.names_keys)})"
                )
            if inplace:
                self.names = [_ for _ in self.names[:-loc]]
                self.names_keys = {
                    _name: _val
                    for _name, _val in self.names_keys.items()
                    if _name in self.names
                }
            else:
                _names = [_ for _ in self.names[:-loc]]
                _names_keys = {
                    _name: _val
                    for _name, _val in self.names_keys.items()
                    if _name in _names
                }
                _dictionary = copy.deepcopy(self.dictionary)
                return MultilayerDict(_names_keys, _dictionary)

        else:
            raise NotImplementedError

    def add_names(self, names_keys: Dict, inplace: bool = False):
        _new_names_keys = dict(**self.names_keys, **names_keys)
        for _keys in itertools.product(*_new_names_keys.values()):
            if not self.keys_exist(_keys):
                raise KeyError(f"The key does not exist {_keys}.")
        if inplace:
            self.names_keys = _new_names_keys
            self.names = list(_new_names_keys.keys())
        else:
            return MultilayerDict(_new_names_keys, self.dictionary)

    def extend(self, keys: List[List[int]], mds: List, inplace: bool = False):
        if len(keys) != np.prod([len(_keys) for _keys in self.names_keys.values()]):
            raise ValueError(
                "The number of MultilayerDict and size of object do not match."
            )
        if len(keys) != len(mds):
            raise ValueError("number of keys and mds do not match.")

        if inplace:
            # dictionary
            for _keys, _md in zip(keys, mds):
                self.update(_keys, _md.dictionary)
            # namesを延長
            self.names.extend(mds[0].names)
            # keysを延長
            self.names_keys.update(mds[0].names_keys)
        else:
            md_return = copy.deepcopy(self)
            # dictionary
            for _keys, _md in zip(keys, mds):
                md_return.update(_keys, _md.dictionary)
            # namesを延長
            md_return.names.extend(mds[0].names)
            # keysを延長
            md_return.names_keys.update(mds[0].names_keys)
            return md_return


def swap_keys(old_multi_dict, new_names: list):
    def index_to_num(index_lists):
        def _make_index_list(_index_list):
            return {_index: _num for _num, _index in enumerate(_index_list)}

        _perm_dict = _make_index_list(index_lists[0])
        return [
            [_perm_dict[_index] for _index in _index_list]
            for _index_list in index_lists
        ]

    def lists_to_permutation(two_row_perm: list):
        if set(two_row_perm[0]) != set(two_row_perm[1]):
            raise ValueError("Permutations do not match.")
        _two_row_perm = dict(zip(*two_row_perm))
        _cycles = []
        _done = set()
        for _i in _two_row_perm.keys():
            if _i not in _done:
                _cycle = [_i]
                _next_elem = _two_row_perm[_i]
                _done.add(_next_elem)
                while _next_elem != _i:
                    _cycle.append(_next_elem)
                    _next_elem = _two_row_perm[_next_elem]
                    _done.add(_next_elem)
                _cycles.append(_cycle)
        return _cycles

    if not (set(new_names) == set(old_multi_dict.names)):
        diff = set(new_names).symmetric_difference(set(old_multi_dict.names))
        raise KeyError(f"Keys {diff} do not match.")

    _two_row_perm = index_to_num([old_multi_dict.names, new_names])
    _cyclic_perm = lists_to_permutation(_two_row_perm)
    _perm = Permutation(_cyclic_perm)

    _new_keys_names = dict(
        zip(
            _perm(list(old_multi_dict.names_keys.keys())),
            _perm(list(old_multi_dict.names_keys.values())),
        )
    )
    _new_multilayer_dict = MultilayerDict(_new_keys_names)
    for _index_keys in itertools.product(*old_multi_dict.names_keys.values()):
        _new_multilayer_dict.update(_perm(_index_keys), old_multi_dict.loc(_index_keys))
    return _new_multilayer_dict


def read_csv(
    file_path: str,
    names_keys: Dict,
    encoding: str = "utf8",
    newline: str = "",
    delimiter: str = ",",
    quotechar: str = "|",
):
    if len(names_keys) == 2:
        with open(file_path, mode="r", encoding=encoding, newline=newline) as _f:
            _dict = dict(
                zip(
                    list(names_keys.values())[0],
                    csv.DictReader(
                        _f,
                        delimiter=delimiter,
                        quotechar=quotechar,
                        fieldnames=list(names_keys.values())[1],
                    ),
                )
            )
        return MultilayerDict(names_keys, _dict)
    elif len(names_keys) == 1:
        with open(file_path, mode="r", encoding=encoding, newline=newline) as _f:
            _dict = list(
                csv.DictReader(
                    _f,
                    delimiter=delimiter,
                    quotechar=quotechar,
                )
            )[0]
        return MultilayerDict(names_keys, _dict)
    else:
        raise NotImplementedError("length of names_keys must be 1 or 2 ")
