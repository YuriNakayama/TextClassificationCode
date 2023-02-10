# # Import

# +
import copy
import itertools
import os

from sympy.combinatorics import Permutation
# -

# # Function


# ## multilayer dict

class multilayer_dict:
    def __init__(self, names_keys: dict):
        self.dict = self.make_multilayer_dict(list(names_keys.values()))
        self.names = list(names_keys.keys())
        self.names_keys = names_keys

    def make_multilayer_dict(self, keys: list):
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

    def loc(self, key_list: list):
        def _loc_recursive(_val, _key_list: list):
            if not _key_list:
                return _val
            else:
                return _loc_recursive(_val[_key_list[0]], _key_list[1:])

        return _loc_recursive(self.dict, key_list)

    def update(self, key_list: list, val):
        _d = self.dict
        for key in key_list[:-1]:
            _d = _d[key]
        _d[key_list[-1]] = val


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
    _new_multilayer_dict = multilayer_dict(_new_keys_names)
    for _index_keys in itertools.product(*old_multi_dict.names_keys.values()):
        _new_multilayer_dict.update(_perm(_index_keys), old_multi_dict.loc(_index_keys))
    return _new_multilayer_dict

