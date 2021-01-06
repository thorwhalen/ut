"""Special dicts"""
__author__ = 'thor'

from collections import defaultdict, UserDict
from ut.pdict.get import set_value_in_nested_key_path

val_unlikely_to_be_value_of_dict = (1987654321, 8239080923)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


class DictDefaultDict(dict):
    """
    Acts similarly to collections.defaultdict, except
        (1) the defaults depend on the key (given by a dict of key-->default_val at construction)
        (2) it is not a function that is called to create the default value (so careful with referenced variables)
    """

    def __init__(self, default_dict):
        super(DictDefaultDict, self).__init__()
        self.default_dict = default_dict

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            return self.default_dict[item]


class KeyPathDict(dict):
    """
    NOTE: Might want to check out key_path.py (in https://github.com/i2mint/py2mint/) instead.

    A dict where you can get and set values from key_paths (i.e. dot-separated strings or lists of nested keys).
    Use with care.
    Some functionalities that would be expected from such a subclass of dict aren't implemented yet, or only partially.

    Further, operating with KeyPathDict is slower. One test showed that getting a value was 80 times slower
    But, to be fair, it was in micro-seconds instead of nano-seconds, so this class can still be useful for
    convenience when it is not in a bottle neck of a process.
    >>> input_dict = {
    ...     'a': {
    ...         'b': 1,
    ...         'c': 'val of a.c',
    ...         'd': [1, 2]
    ...     },
    ...     'b': {
    ...         'A': 'val of b.A',
    ...         'B': {
    ...             'AA': 'val of b.B.AA'
    ...         }
    ...     },
    ...     10: 'val for 10',
    ...     '10': 10
    ... }
    >>>
    >>> d = KeyPathDict(input_dict)
    >>> d
    {'a': {'b': 1, 'c': 'val of a.c', 'd': [1, 2]}, 'b': {'A': 'val of b.A', 'B': {'AA': 'val of b.B.AA'}}, 10: 'val for 10', '10': 10}
    >>> d.get('a.c')
    'val of a.c'
    >>> d.get(['a', 'c']) == d['a.c']
    True
    >>> d[['a', 'c']] == d['a.c']
    True
    >>> d.get('non.existent.key', 'default')
    'default'
    >>> d['b.B.AA']
    'val of b.B.AA'
    >>> d['b.B.AA'] = 3  # assigning another value to EXISTING key path
    >>> d['b.B.AA']
    3
    >>> d['10'] = 0  # assigning another value to EXISTING key path
    >>> d['10']
    0
    >>> d['new_key'] = 7  # assigning another value to new SINGLE key
    >>> d['new_key']
    7
    >>> d['new.key.path'] = 8  # assigning a value to new key path
    >>> d['new.key']
    {'path': 8}
    >>> d['new.key.old.path'] = 9  # assigning a value to new key path, intersecting with another
    >>> d['new.key']
    {'path': 8, 'old': {'path': 9}}
    >>> d['new.key'] = 'something new'  # assigning a value to a key (sub-)path that already exists
    >>> d['new.key']
    'something new'
    """

    def get(self, key_path, d=None):
        #         return get_value_in_key_path(dict(KeyPathDict), key_path, d)
        if isinstance(key_path, str):
            key_path = key_path.split('.')
        if isinstance(key_path, list):
            k_length = len(key_path)
            if k_length == 0:
                return super(KeyPathDict, self).get(key_path[0], d)
            else:
                val_so_far = super(KeyPathDict, self).get(key_path[0], d)
                for key in key_path[1:]:
                    if isinstance(val_so_far, dict):
                        val_so_far = val_so_far.get(key, val_unlikely_to_be_value_of_dict)
                        if val_so_far == val_unlikely_to_be_value_of_dict:
                            return d
                    else:
                        return d
                return val_so_far
        else:
            return super(KeyPathDict, self).get(key_path, d)

    def __getitem__(self, val):
        return self.get(val, None)

    def __setitem__(self, key_path, val):
        """
        Only works with EXISTING key_paths or SINGLE keys
        :param key_path:
        :param val:
        :return:
        """
        if isinstance(key_path, str):
            key_path = key_path.split('.')
        if isinstance(key_path, list):
            first_key = key_path[0]
            if len(key_path) == 1:
                super(KeyPathDict, self).__setitem__(first_key, val)
                # self[first_key] = val
            else:
                if first_key in self:
                    set_value_in_nested_key_path(self[first_key], key_path[1:], val)
                else:
                    self[first_key] = {}
                    set_value_in_nested_key_path(self[first_key], key_path[1:], val)
        else:
            super(KeyPathDict, self).__setitem__(key_path, val)

    def __contains__(self, key_path):
        if isinstance(key_path, str):
            key_path = key_path.split('.')
        if isinstance(key_path, list):
            if len(key_path) == 1:
                return super(KeyPathDict, self).__contains__(key_path[0])
            else:
                tmp = super(KeyPathDict, self).__getitem__(key_path[0])
                for k in key_path[1:]:
                    if not isinstance(tmp, dict) or k not in tmp:
                        return False
                    tmp = tmp[k]
                return True
        else:
            return super(KeyPathDict, self).__contains__(key_path)
