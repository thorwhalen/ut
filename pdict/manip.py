__author__ = 'thorwhalen'

from collections import MutableMapping
from itertools import chain, imap

from ut.pdict.get import iter_key_path_items, set_value_in_nested_key_path


def transform_dict(d, key_path_trans):
    """
    Make a transformed copy of a dict.
    :param d: a dict (to copy and transform)
    :param key_path_trans: a one-level dict mapping key_paths (possibly ending with a wildcard '*') to an instruction
    of what to do with this item. This instruction could be:
        * the string "ignore_entry", which will result in the item being ignored (like poping it from the original dict)
        * any string (will result in the value of that item being copied to a (key path) field baring that name)
        * a function, which will be applied to the value of the item
    :return:
    >>> key_path_trans = {
    ...     'a.b': int,
    ...     'a.c': 'new_ac',
    ...     'b.*': lambda x: x * 10,
    ...     'delete_this': 'ignore_entry'
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
    ...     'delete_this': 'should be ommited'
    ... }
    >>> transform_dict(input_dict, key_path_trans)
    {'a': {'b': 3, 'd': 'this should remain as is'}, 'new_ac': 'I am a.c', 'b': {'A': 10, 'B': 20}}
    """
    new_d = dict()
    wildcard_prefixes = map(lambda k: k[:-1], filter(lambda k: k.endswith('*'), key_path_trans.keys()))

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
            elif trans_func == 'ignore_entry':
                continue  # skip this one (so you won't have it in new_d)
            elif trans_func == 'entry_asis':
                set_value_in_nested_key_path(new_d, key_path, val)  # take value as is
            else:
                if isinstance(trans_func, basestring):  # assume trans_func is a field name...
                    set_value_in_nested_key_path(new_d, trans_func, val)  # ... which we want to rename key_path by.
                else:
                    raise TypeError("trans_func must be a callable or a string")
        else:
            set_value_in_nested_key_path(new_d, key_path, val)  # take value as is

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
        return [dict({prefix + k: v for k, v in element.iteritems()}, **d) for element in key_value]
    else:
        return list(chain(*imap(lambda x: rollout(x, key=key, sep=sep, copy=copy), d)))


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
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
