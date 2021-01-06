"""Make filters with the mongo language"""
import operator
from functools import partial

from ut.pdict.get import get_value_in_key_path

print("This is a partial implementation of mongoquery (https://pypi.org/project/mongoquery/) "
      "before I knew it existed. Might want to use mongoquery instead.")


def nor(*args):
    for a in args:
        if not bool(a):
            return True
    return False


mg_op_key_to_op = {
    '$gte': operator.le,  # operator reversed because arguments inverse of order we need in code
    '$gt': operator.lt,  # operator reversed because arguments inverse of order we need in code
    '$lte': operator.ge,  # operator reversed because arguments inverse of order we need in code
    '$lt': operator.gt,  # operator reversed because arguments inverse of order we need in code
    '$in': operator.contains,
    '$nin': lambda element, _set: not operator.contains(_set, element),
    '$eq': operator.eq,
    '$neq': operator.ne,
    '$not': operator.not_,
    '$and': operator.__and__,  # TODO: Need to extend to more than two operands
    '$or': operator.__or__,  # TODO: Need to extend to more than two operands
    '$nor': nor
}  # NOTE: Apparent misalignment of gt and lt ops on purpose (order of operator and use is flipped.


def mg_filt_kv_to_func_2(key_path, val_condition):
    """
    """
    func_list = _func_list_of_val_condition(val_condition)
    if func_list:
        if isinstance(val_condition, dict):
            if func_list:
                return _conjunction_key_func_for_func_list_and_key_path(func_list, key_path)
        elif isinstance(val_condition, list):
            if key_path == '$and':
                return _conjunction_key_func_for_func_list_and_key_path(func_list, key_path)
            elif key_path == '$or':
                return _disjunction_key_func_for_func_list_and_key_path(func_list, key_path)
            else:
                raise ValueError("val_condition was a list, but key_path was neither $and nor $or")

    def key_func(_dict):
        return get_value_in_key_path(_dict, key_path, None) == val_condition

    return key_func


def _conjunction_key_func_for_func_list_and_key_path(func_list, key_path):
    def key_func(_dict):
        """
        Returns True if and only iff all func_list funcs return True
        """
        for f in func_list:
            if not f(get_value_in_key_path(_dict, key_path, None)):
                return False  # stop early, we know it's False
        return True

    return key_func


def _disjunction_key_func_for_func_list_and_key_path(func_list, key_path):
    def key_func(_dict):
        """
        Returns True if and only iff all func_list funcs return True
        """
        for f in func_list:
            if not f(get_value_in_key_path(_dict, key_path, None)):
                return True  # stop early, we know it's False
        return False

    return key_func


def _func_list_of_val_condition(val_condition):
    func_list = list()
    if isinstance(val_condition, dict):
        for k, v in val_condition.items():
            if k.startswith('$'):
                func_list.append(partial(mg_op_key_to_op[k], v))
    return func_list


def mg_filt_kv_to_func(key_path, val_condition):
    """
    Returns a function that checks
        (1) The existance of a key path in a dict
        (2) If the value satisfies the condition specified by mongo-query-like val_condition
    :param key_path: A key path. That is, a field name, or a nested field specification,
        such as 'this.is.a.nested.field'.
    :param val_condition: A value or a (single field) mongo-query-like dict
    :return: A function
    """
    if isinstance(val_condition, dict):
        func_list = list()
        for k, v in val_condition.items():
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
    """
    Just applies mg_filt_kv_to_func to every element of the (dict) mg_filt
    :param mg_filt: A mongo-query-like dict (can have several fields
    :return: A list of functions
    """
    return [mg_filt_kv_to_func(*x) for x in iter(mg_filt.items())]


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

    >>> ####### A complicated one
    >>> mg_filt = {
    ...    'a': {'$in': [3, 4, 5]},
    ...    'x': {'$gte': 10, '$lt': 20},
    ...    'foo.bar': 'bit',
    ...    'this': {'is': 'that'},
    ... }
    >>> filt = dict_filt_from_mg_filt(mg_filt)
    >>> filt({'a': 3, 'x': 15, 'foo': {'bar': 'bit'}, 'this': {'is': 'that'}, 'and_something': 'else'})
    True
    >>> filt({'a': 1, 'x': 15, 'foo': {'bar': 'bit'}, 'this': {'is': 'that'}, 'and_something': 'else'})
    False
    >>> filt({'a': 3, 'x': 20, 'foo': {'bar': 'bit'}, 'this': {'is': 'that'}, 'and_something': 'else'})
    False
    >>> filt({'a': 3, 'x': 15, 'foo.bar': 'bit', 'this': {'is': 'that'}, 'and_something': 'else'})
    False
    >>> ####### testing equality
    >>> filt = dict_filt_from_mg_filt(mg_filt={'foo': 'bar'})
    >>> # True when equal
    >>> filt({'foo': 'bar'})
    True
    >>> # false when not equal
    >>> filt({'foo': 'bear'})
    False
    >>> # false if key not present
    >>> filt({'fool': 'bar'})
    False
    >>> # can also have equality of dicts
    >>> filt = dict_filt_from_mg_filt(mg_filt={'foo': {'bar': 'bit'}})
    >>> filt({'foo': {'bar': 'bit'}})
    True
    >>> ####### A single >= comparison
    >>> mg_filt = {'a': {'$gte': 10}}
    >>> filt = dict_filt_from_mg_filt(mg_filt)
    >>> filt({'a': 9})
    False
    >>> filt({'a': 10})
    True
    >>> filt({'a': 11})
    True
    >>> ####### A single > comparison
    >>> filt = dict_filt_from_mg_filt({'a': {'$gt': 10}})
    >>> filt({'a': 9})
    False
    >>> filt({'a': 10})
    False
    >>> filt({'a': 11})
    True
    >>> ####### A range query
    >>> filt = dict_filt_from_mg_filt({'a': {'$gte': 10, '$lt': 20}})
    >>> map(filt, [{'a': x} for x in [9, 10, 15, 20, 21]])
    [False, True, True, False, False]
    """
    key_func_list = key_func_list_from_mg_filt(mg_filt)

    def _filt(_dict):
        for i, f in enumerate(key_func_list):
            if not f(_dict):
                return False
        return True  # if you get this far

    return _filt
