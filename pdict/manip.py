"""Manipulating dicts"""
__author__ = 'thorwhalen'

from collections.abc import MutableMapping
from itertools import chain

from ut.pdict.get import iter_key_path_items, set_value_in_nested_key_path

ASIS = '_keep_val_as_is'
DROP = '_drop_key_path_entry'


def transform_dict(d, key_path_trans, keep_unspecified_key_paths=True):
    """
    Make a transformed copy of a dict.
    :param d: a dict (to copy and transform)
    :param key_path_trans: a one-level dict mapping key_paths (possibly ending with a wildcard '*') to an instruction
    of what to do with this item. This instruction could be:
        * the string '_drop_key_path_entry' (better used through the DROP var that can be imported from this module)
            which will result in the item being ignored (like popping it from the original dict)
        * the string '_keep_val_as_is' (better used through the ASIS var that can be imported from this module)
            which will result in the item being kept as is.
        * any string (will result in the value of that item being copied to a (key path) field baring that name)
        * a function, which will be applied to the value of the item
        * a (new_key_path, function) tuple: the function will be applied to the value, the result placed in new_key_path
    :param keep_unspecified_key_paths: Whether to keep unspecified key paths (same as specifying "entry_asis" asis), or ignore them
    :return:
    >>> key_path_trans = {
    ...     'a.b': int,
    ...     'a.c': 'new_ac',
    ...     'b.*': lambda x: x * 10,
    ...     'delete_this': '_drop_key_path_entry'
    ... }
    >>> input_dict = {
    ...     'a': {
    ...         'b': 3.14,  # should become 3 (int)
    ...         'c': 'I am a.c',  # should move to new_ac field
    ...         'd': 'this should remain as is'
    ...     },
    ...     'b': {
    ...         'A': 1,
    ...         'B': 2
    ...     },
    ...     'delete_this': 'should be omited'
    ... }
    >>> transform_dict(input_dict, key_path_trans)
    {'a': {'b': 3, 'd': 'this should remain as is'}, 'new_ac': 'I am a.c', 'b': {'A': 10, 'B': 20}}
    """
    new_d = dict()
    wildcard_prefixes = [k[:-1] for k in [k for k in list(key_path_trans.keys()) if k.endswith('*')]]

    def to_trans_key(key_path):
        for prefix in wildcard_prefixes:
            if key_path.startswith(prefix):
                return prefix + '*'
        return key_path

    for key_path, val in iter_key_path_items(d):
        trans_key = to_trans_key(key_path)
        if trans_key in key_path_trans:
            trans_func = key_path_trans[trans_key]
            if callable(trans_func):
                set_value_in_nested_key_path(new_d, key_path, trans_func(val))  # apply trans_func to val
            elif trans_func == DROP:
                continue  # skip this one (so you won't have it in new_d)
            elif trans_func == ASIS:
                set_value_in_nested_key_path(new_d, key_path, val)  # take value as is
            elif isinstance(trans_func, (tuple, list)) and len(trans_func) == 2:
                new_key_path = trans_func[0]  # the key path where we should store the value
                trans_func = trans_func[1]
                set_value_in_nested_key_path(new_d, new_key_path, trans_func(val))  # apply trans_func to val
            else:
                if isinstance(trans_func, str):  # assume trans_func is a field name...
                    set_value_in_nested_key_path(new_d, trans_func, val)  # ... which we want to rename key_path by.
                else:
                    raise TypeError("trans_func must be a callable or a string")
        else:  # if trans_key is not listed in key_path_trans keys...
            if keep_unspecified_key_paths:  # then consider it as a "_keep_key_path"
                set_value_in_nested_key_path(new_d, key_path, val)  # take value as is
            # else will ignore it (same effect as "ignore_entry"

    return new_d


def rollout(d, key, sep='.', copy=True):
    if isinstance(d, dict):
        if copy:
            d = d.copy()
        key_value = d.pop(key)
        if sep is None:
            prefix = ''
        else:
            prefix = key + sep
        return [dict({prefix + k: v for k, v in element.items()}, **d) for element in key_value]
    else:
        return list(chain(*map(lambda x: rollout(x, key=key, sep=sep, copy=copy), d)))


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in list(d.items()):
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(list(flatten(v, new_key, sep=sep).items()))
        else:
            items.append((new_key, v))
    return dict(items)


def add_defaults(d, default_dict):
    return dict(default_dict, **d)


def recursive_left_union(a, b):
    b_copy = b.copy()
    recursively_update_with(b_copy, a)
    return b_copy


def recursively_update_with(a, b):
    """
    Recursively updates a with b.
    That is:
        It works like the standard a.update(b) when the values are not both dicts (that is, it overwrite a[key] with
        b[key] whether key is in a or not, but does the updating inside the dicts themselves if a[key] and b[key] are
        both dicts.
    Example:
    a = {0: 0, 1: 1, 2:{'a': 'a', 'b': 'b'}}
    b = {1: 11, 2:{'b': 'bb', 'c': 'cc'}, 3: 33}
    recursively_update_with(a, b)
    assert a == {0: 0, 1: 11, 2: {'a': 'a', 'b': 'bb', 'c': 'cc'}, 3: 33}

>>> a = {0: 0, 1: 1, 2:{'a': 'a', 'b': 'b'}}
>>> b = {1: 11, 2:{'b': 'bb', 'c': 'cc'}, 3: 33}
>>> recursively_update_with(a, b)
>>> assert a == {0: 0, 1: 11, 2: {'a': 'a', 'b': 'bb', 'c': 'cc'}, 3: 33}
    """

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                recursively_update_with(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]


def merge(a, b, path=None):
    "merges b into a"
    a = a.copy()
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
