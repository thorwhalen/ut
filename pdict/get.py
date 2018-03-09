__author__ = 'thorwhalen'

import ut.util.ulist as ulist
from numpy import isnan
from functools import reduce  # forward compatibility for Python 3
import operator
from functools import partial


# def recursive_list_of_keys(d, key=None):
#     """
#     recursive list of lists of dict keys:
#         get a list (of lists of lists...) of keys of the dict (and the dict's dicts...)
#     """
#     try:
#         if isinstance(d, list) and len(d[0]) > 0 and isinstance(d[0], dict):
#
#         klist = d.keys()
#         for i, x in enumerate(klist):
#             klist[i] = recursive_list_of_keys(d[x])
#         return klist
#     except BaseException:
#         return d

def nor(*args):
    for a in args:
        if not bool(a):
            return True
    return False


mg_op_key_to_op = {
    '$gte': operator.le,
    '$gt': operator.lt,
    '$lte': operator.ge,
    '$lt': operator.gt,
    '$in': operator.contains,
    '$nin': lambda element, _set: not operator.contains(_set, element),
    '$eq': operator.eq,
    '$neq': operator.ne,
    '$not': operator.not_,
    '$and': operator.__and__,  # TODO: Need to extend to more than two operands
    '$or': operator.__or__,  # TODO: Need to extend to more than two operands
    '$nor': nor
}  # NOTE: Apparent misalignment of gt and lt ops on purpose (order of operator and use is flipped.


# TODO: Might want $exists and $type?

def mg_filt_kv_to_func(key_path, val_condition):
    if isinstance(val_condition, dict):
        func_list = list()
        for k, v in val_condition.iteritems():
            if k.startswith('$'):
                func_list.append(partial(mg_op_key_to_op[k], v))
        if func_list:
            def key_func(_dict):
                """
                Returns True if and only iff all func_list funcs return True
                """
                for f in func_list:
                    if not f(get_value_in_key_path(_dict, key_path, None)):
                        return False  # stop early, we know it's False
                return True

            return key_func

    def key_func(_dict):
        return get_value_in_key_path(_dict, key_path, None) == val_condition

    return key_func


def key_func_list_from_mg_filt(mg_filt):
    return map(lambda x: mg_filt_kv_to_func(*x), mg_filt.iteritems())


def dict_filt_from_mg_filt(mg_filt):
    """
    The final intent is to do what a mongo query does on a collection, but with a dict iterator instead.
    dict_filt_from_mg_filt creates, based on a mg_filt (that uses the same language as mongo, a filter.
    A filter is a function that returns True iff mg_filt condition is satistfied.
    Basically, if mgc is a mongo collection and dict_iter is a dict iterator containing the same dicts,
    the following should be equivalent:
        mgc.find(mg_filt) and itertools.ifilter(dict_filt_from_mg_filt(mg_filt), dict_iter)
    In fact, you can test this directly with a (small) mongo collection by doing:
        assert list(mgc.find(mg_filt)) == filter(dict_filt_from_mg_filt(mg_filt), mgc.find())
    :param mg_filt:
    :return: a filter (a function returning True or False)

    >>> mg_filt = {'a': {'$gt': 10}}
    >>> filt = dict_filt_from_mg_filt()
    """
    key_func_list = key_func_list_from_mg_filt(mg_filt)

    def _filt(_dict):
        for i, f in enumerate(key_func_list):
            if not f(_dict):
                return False
        return True  # if you get this far

    return _filt


def iter_key_path_items(d, key_path_prefix=None):
    """
    iterate through items of dict recursively, yielding (key_path, val) pairs for all nested values that are not dicts.
    That is, if a value is a dict, it won't generate a yield, but rather, will be iterated through recursively.
    :param d: input dict
    :param key_path_so_far: string to be prepended to all key paths (for use in recursion, not meant for direct use)
    :return: a (key_path, val) iterator
    >>> input_dict = {
    ...     'a': {
    ...         'a': 'a.a',
    ...         'b': 'a.b',
    ...         'c': {
    ...             'a': 'a.c.a'
    ...         }
    ...     },
    ...     'b': 'b',
    ...     'c': 3
    ... }
    >>> list(iter_key_path_items(input_dict))
    [('a.a', 'a.a'), ('a.c.a', 'a.c.a'), ('a.b', 'a.b'), ('c', 3), ('b', 'b')]
    """
    if key_path_prefix is None:
        for k, v in d.iteritems():
            if not isinstance(v, dict):
                yield k, v
            else:
                for kk, vv in iter_key_path_items(v, k):
                    yield kk, vv
    else:
        for k, v in d.iteritems():
            if not isinstance(v, dict):
                yield key_path_prefix + '.' + k, v
            else:
                for kk, vv in iter_key_path_items(v, k):
                    yield key_path_prefix + '.' + kk, vv


def extract_key_paths(d, key_paths, field_naming='full', use_default=False, default_val=None):
    """
    getting with a key list or "."-separated string
    :param d: dict
    :param key_path: list or "."-separated string of keys
    :param field_naming: 'full' (default) will use key_path strings as is, leaf will only use the last dot item
        (i.e. this.is.a.key.path will result in "path" being used)
    :return:
    """
    dd = dict()
    if isinstance(key_paths, dict):
        key_paths = [k for k, v in key_paths.iteritems() if v]

    for key_path in key_paths:

        if isinstance(key_path, basestring):
            field = key_path
            key_path = key_path.split('.')
        else:
            field = '.'.join(key_path)

        if field_naming == 'leaf':
            field = key_path[-1]
        else:
            field = field

        try:
            dd.update({field: reduce(operator.getitem, key_path, d)})
        except (TypeError, KeyError):
            if use_default:
                dd.update({field: default_val})

    return dd


def key_paths(d):
    key_path_list = list()
    for k, v in d.iteritems():
        if not isinstance(v, dict):
            key_path_list.append(k)
        else:
            key_path_list.extend(map(lambda x: k + '.' + x, key_paths(v)))
    return key_path_list


def get_value_in_key_path(d, key_path, default_val=None):
    """
    getting with a key list or "."-separated string
    :param d: dict
    :param key_path: list or "."-separated string of keys
    :return:
    """
    if isinstance(key_path, basestring):
        key_path = key_path.split('.')
    try:
        return reduce(operator.getitem, key_path, d)
    except (TypeError, KeyError):
        return default_val


def set_value_in_key_path(d, key_path, val):
    """
    setting with a key list or "."-separated string
    :param d: dict
    :param key_path: list or "."-separated string of keys
    :param val: value to assign
    :return:
    """
    if isinstance(key_path, basestring):
        key_path = key_path.split('.')
    get_value_in_key_path(d, key_path[:-1])[key_path[-1]] = val


def set_value_in_nested_key_path(d, key_path, val):
    """

    :param d:
    :param key_path:
    :param val:
    :return:
    >>> input_dict = {
    ...   "a": {
    ...     "c": "val of a.c",
    ...     "b": 1,
    ...   },
    ...   "10": 10,
    ...   "b": {
    ...     "B": {
    ...       "AA": 3
    ...     }
    ...   }
    ... }
    >>>
    >>> set_value_in_nested_key_path(input_dict, 'new.key.path', 7)
    >>> input_dict
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': {'path': 7}}}
    >>> set_value_in_nested_key_path(input_dict, 'new.key.old.path', 8)
    >>> input_dict
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': {'path': 7, 'old': {'path': 8}}}}
    >>> set_value_in_nested_key_path(input_dict, 'new.key', 'new val')
    >>> input_dict
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}}, 'new': {'key': 'new val'}}
    """
    if isinstance(key_path, basestring):
        key_path = key_path.split('.')
    first_key = key_path[0]
    if len(key_path) == 1:
        d[first_key] = val
    else:
        if first_key in d:
            set_value_in_nested_key_path(d[first_key], key_path[1:], val)
        else:
            d[first_key] = {}
            set_value_in_nested_key_path(d[first_key], key_path[1:], val)


def mk_fixed_coordinates_value_getter(get_key_list):
    return \
        lambda the_dict: \
            reduce(lambda x, y: x.get(y, {}), get_key_list, the_dict) or None


def key_if_exists_else_return_none(d, key):
    DeprecationWarning('You should really call this one liner directly!!')
    return d.get(key, None)


def head(d, num_of_elements=5, start_at=0):
    """
    get the "first" few (num) elements of a dict
    """
    return {k: d[k] for k in d.keys()[start_at:min(len(d), start_at + num_of_elements)]}


def tail(d, num_of_elements=5):
    """
    get the "first" few (num) elements of a dict
    """
    return {k: d[k] for k in d.keys()[-min(len(d), num_of_elements):]}


def left_union(d, defaults):
    """
    :param d: dict
    :param defaults: dict
    :return: dict d, enhanced with key:value pairs of defaults dict whose keys weren't found in d
    """
    return dict(defaults, **d)


def get_subset_of_defaults(d, defaults, subset_of_default_keys):
    """
    :param d: dict
    :param defaults: dict
    :param subset_of_default_keys: list of keys
    :return: adds key:value pairs to d if key is not in d, but is in defaults and subset_of_default_keys
    """
    return left_union(d, get_subdict(defaults, subset_of_default_keys))


def get_subdict(d, list_of_keys):
    """
    :param d: dict
    :param subset_of_keys: list of keys
    :return: the subset of key:value pairs of d where key is in list_of_keys
    """
    return dict([(i, d[i]) for i in list_of_keys if i in d])


def get_subdict_and_remainder(d, list_of_keys):
    """
    :param d: dict
    :param subset_of_keys: list of keys
    :return: the subset of key:value pairs of d where key is in list_of_keys
    """
    keys_in = set(d.keys()).intersection(list_of_keys)
    keys_not_in = set(d.keys()).difference(list_of_keys)
    return (dict([(i, d[i]) for i in keys_in]), dict([(i, d[i]) for i in keys_not_in]))


def all_but(d, exclude_keys):
    return get_subdict(d, set(d.keys()).difference(ulist.ascertain_list(exclude_keys)))


def all_non_null(d):
    return {k: v for k, v in d.iteritems() if v is not None and not isnan(v)}
