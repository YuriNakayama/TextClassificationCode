# # Import

# +
import copy
import itertools
import os
from typing import List, Dict

from sympy.combinatorics import Permutation
# -

# # Function


# ## multilayer dict

class MultilayerDict:
    def __init__(
        self,
        names_keys: Dict | None = None,
        dir_path: str | None = None,
        names: List | None = None,
        extension: str = "csv",
        basis_depth: str = "max",
    ):
        if names_keys is not None:
            self.dict = self.make_multilayer_dict(list(names_keys.values()))
            self.names = list(names_keys.keys())
            self.names_keys = names_keys
        elif dir_path is not None:
            self.dict, _keys, self.names = self.read_dirs(
                dir_path,
                names=names,
            )
        else:
            raise ValueError("Both names_keys and dir_path are None.")

    def make_multilayer_dict(self, keys: List):
        def _multilayer_dict_recursive(_d: dict, _keys: list):
            if not _keys:
                return _d, []
            else:
                return _multilayer_dict_recursive(
                    {_key: copy.deepcopy(_d) for _key in _keys[-1]}, _keys[:-1]
                )

        _multilayer_dict, _ = _multilayer_dict_recursive(dict(), keys)
        return _multilayer_dict

    def name_is_in(self, name):
        return name in self.name

    def loc(self, key_list: List):
        def _loc_recursive(_val, _key_list: list):
            if not _key_list:
                return _val
            else:
                return _loc_recursive(_val[_key_list[0]], _key_list[1:])

        return _loc_recursive(self.dict, key_list)

    def update(self, key_list: List, val):
        _d = self.dict
        for key in key_list[:-1]:
            _d = _d[key]
        _d[key_list[-1]] = val

    def _one_dimensional_csv_to_dict(
        file_path, encoding="utf8", newline="", delimiter=",", quotechar="|"
    ):
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

    def read_dirs(self, dir_path, names=None, extension="csv", basis_depth="max"):
        def _get_uniform_dirs(_path_dirs, _basis_depth):
            # 最も深いパスを基準にMultilayerDictを生成する
            if _basis_depth == "max":
                _depth = max(map(len, _path_dirs.values()))
                _uniform_dirs = {
                    _path: _dir
                    for _path, _dir in _path_dirs.items()
                    if len(_dir) == _depth
                }
                return _depth, _unifrom_dirs
            elif _basis_depth == "min":
                _depth = min(map(len, _path_dirs.values()))
                _uniform_dirs = {
                    _path: _dir
                    for _path, _dir in _path_dirs.items()
                    if len(_dir) == _depth
                }
                return _depth, _unifrom_dirs
            elif isinstance(_basis_depth, int) & (_basis_depth > 0):
                _depth = _basis_depth
                _uniform_dirs = {
                    _path: _dir
                    for _path, _dir in _path_dirs.items()
                    if len(_dir) == _depth
                }
                return _depth, _unifrom_dirs
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

        _path_dirs, _depth = _get_uniform_dirs(_dirs, basis_depth)

        _md = make_multilayer_dict(_names)
        # ファイルの読み込み
        _file_keys = set()
        if extension == "csv":
            for _path, _dir_list in _path_dirs.items():
                _file_dict = _one_dimensional_csv_to_dict(_path)
                md.update(_dir_list, _file_dict)
                _file_keys = _file_keys & set(_file_dict)
        elif extension == "json":
            for _path, _dir_list in _path_dirs.items():
                _file_dict = _one_dimensional_json_to_dict(_path)
                md.update(_dir_list, _file_dict)
                _file_keys = _file_keys & set(_file_dict)
        else:
            raise NotImplementedError

        _keys = [list(set(_)) for _ in zip(*_path_dirs.values())]
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

        return _md, _keys, _names


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




