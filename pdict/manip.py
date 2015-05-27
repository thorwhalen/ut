__author__ = 'thorwhalen'


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
