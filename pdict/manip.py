__author__ = 'thorwhalen'

from collections import MutableMapping
from numpy import concatenate
from itertools import chain, imap


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
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
