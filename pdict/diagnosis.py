"""Diagnosing dictionaries"""
__author__ = 'thorwhalen'

import pprint
import numpy as np
import pandas as pd
import json
from ut.pdict.get import iter_key_path_items


def print_key_paths(d):
    print('\n'.join(k for k, _ in iter_key_path_items(d)))


def print_key_paths_and_val_peep(d, n_characters_in_val_peep=15):
    if n_characters_in_val_peep is not None:
        print('\n'.join(f"{k}: " + f"{v}..."[:n_characters_in_val_peep] for k, v in iter_key_path_items(d)))
    else:
        print('\n'.join(f"{k}: " + f"{v}" for k, v in iter_key_path_items(d)))


base_validation_funs = {
    "be a": isinstance,
    "be in": lambda val, check_val: val in check_val,
    "be at least": lambda val, check_val: val >= check_val,
    "be more than": lambda val, check_val: val > check_val,
    "be no more than": lambda val, check_val: val <= check_val,
    "be less than": lambda val, check_val: val < check_val,
}


def validate_kwargs(kwargs_to_validate,
                    validation_dict,
                    validation_funs=base_validation_funs,
                    all_kwargs_should_be_in_validation_dict=False,
                    ignore_misunderstood_validation_instructions=False
                    ):
    """
    Utility to validate a dict. It's main use is to validate function arguments (expressing the validation checks
    in validation_dict) by doing validate_kwargs(locals()), usually in the beginning of the function
    (to avoid having more accumulated variables than we need in locals())
    :param kwargs_to_validate: as the name implies...
    :param validation_dict: A dict specifying what to validate. Keys are usually name of variables (when feeding
        locals()) and values are dicts, themselves specifying check:check_val pairs where check is a string that
        points to a function (see validation_funs argument) and check_val is an object that the kwargs_to_validate
        value will be checked against.
    :param validation_funs: A dict of check:check_function(val, check_val) where check_function is a function returning
        True if val is valid (with respect to check_val).
    :param all_kwargs_should_be_in_validation_dict: If True, will raise an error if kwargs_to_validate contains
        keys that are not in validation_dict.
    :param ignore_misunderstood_validation_instructions: If True, will raise an error if validation_dict contains
        a key that is not in validation_funs (safer, since if you mistype a key in validation_dict, the function will
        tell you so!
    :return: True if all the validations passed.

    >>> validation_dict = {
    ...     'system': {
    ...         'be in': {'darwin', 'linux'}
    ...     },
    ...     'fv_version': {
    ...         'be a': int,
    ...         'be at least': 5
    ...     }
    ... }
    >>> validate_kwargs({'system': 'darwin'}, validation_dict)
    True
    >>> try:
    ...     validate_kwargs({'system': 'windows'}, validation_dict)
    ... except AssertionError as e:
    ...     print(e)
    system must be in set(['darwin', 'linux'])
    >>> try:
    ...     validate_kwargs({'fv_version': 9.9}, validation_dict)
    ... except AssertionError as e:
    ...     print(e)
    fv_version must be a <type 'int'>
    >>> try:
    ...     validate_kwargs({'fv_version': 4}, validation_dict)
    ... except AssertionError as e:
    ...     print(e)
    fv_version must be at least 5
    >>> validate_kwargs({'fv_version': 6}, validation_dict)
    True
    """
    validation_funs = dict(base_validation_funs, **validation_funs)
    for var, val in kwargs_to_validate.items():  # for every (var, val) pair of kwargs
        if var in validation_dict:  # if var is in the validation_dict
            for check, check_val in validation_dict[var].items():  # for every (key, val) of this dict
                if check in base_validation_funs:  # if you have a validation check for it
                    if not validation_funs[check](val, check_val):  # check it's valid
                        raise AssertionError("{} must {} {}".format(var, check, check_val))  # and raise an error if not
                elif not ignore_misunderstood_validation_instructions:  # should ignore if check not understood?
                    raise AssertionError("I don't know what to do with the validation check '{}'".format(
                        check
                    ))
        elif all_kwargs_should_be_in_validation_dict:  # should all variables have checks?
            raise AssertionError("{} wasn't in the validation_dict")
    return True


def json_size_of_fields(d):
    """
    Get the json-size of the values of a dict.
    :param d: dict to diagnose
    :return: dict with the same first level fields as the input dict, but with:
        * an integer representing the length of the json.dumps string of the dict's field value
        * None if there was a problem json-izing the value
    """
    diag = dict()
    for k, v in d.items():
        try:
            diag[k] = len(json.dumps(v))
        except:
            diag[k] = None
    return diag


def are_equal_on_common_keys(dict1, dict2):
    """
    Return True if and only if all (nested) values of the dicts are the same for the keys (paths) they share
    :param dict1:
    :param dict2:
    :return:
    """
    for k in dict1:
        if k in dict2:
            v1 = dict1[k]
            v2 = dict2[k]
            if isinstance(v1, dict):
                if isinstance(v2, dict):
                    if not are_equal_on_common_keys(v1, v2):
                        return False
                else:
                    return False
            else:
                if v1 != v2:
                    return False
    return True  # if you've never returned (with False), you're True (your dicts are equal)


def first_difference_on_common_keys(dict1, dict2, key_path_so_far=None):
    """
    Returns an empty list if dicts are equal (in the are_equal_on_common_keys sense).
    If not, returns the first dict path where they differ.
    :param dict1:
    :param dict2:
    :param key_path_so_far:
    :return:
    """
    if key_path_so_far is None:
        key_path_so_far = list()
    for k in dict1:
        if k in dict2:
            v1 = dict1[k]
            v2 = dict2[k]
            if isinstance(v1, dict):
                if isinstance(v2, dict):
                    diff = first_difference_on_common_keys(v1, v2, key_path_so_far + [k])
                    if len(diff) > 0:
                        return diff
                else:
                    return key_path_so_far + [k]
            else:
                if v1 != v2:
                    return key_path_so_far + [k]
    return list()  # if you've never returned (with False), you're True (your dicts are equal)


def typeof(x):
    if isinstance(x, list):
        if len(x) > 0:
            unik_types = list(np.lib.unique([typeof(xx) for xx in x]))
            if len(unik_types) == 1:
                return "list of " + unik_types[0]
            elif len(unik_types) <= 3:
                return "list of " + ", ".join(unik_types)
            else:
                return "list of various types"
        else:
            return "empty list"
    else:
        return type(x).__name__


def print_dict_example(dict_list, recursive=True):
    ppr = pprint.PrettyPrinter(indent=2)
    ppr.pprint(example_dict_from_dict_list(dict_list, recursive=recursive))


def print_dict_example_types(dict_list, recursive=True):
    ppr = pprint.PrettyPrinter(indent=2)
    ppr.pprint(dict_of_types_of_dict_values(dict_list, recursive=recursive))


def example_dict_from_dict_list(dict_list, recursive=False):
    """
    Returns a dict that "examplifies" the input list of dicts
    Indeed, you may have a list of dicts but that don't contain the same keys.
    This function will pick the first new key,value it finds and pack it in a same dict

    If the argument recursive is True, the function will call itself recursively on any values that are lists of dicts

     For example:
        d1 = {'a':1, 'b':2}
        d2 = {'b':20, 'c':30}
        d12 = example_dict_from_dict_list([d1,d2])
        assert d12=={'a':1, 'b':2, 'c':30}
    """
    if not isinstance(dict_list, list):
        if isinstance(dict_list, dict):
            dict_list = [dict_list]
        else:
            raise TypeError("dict_list must be a dict or a list of dicts")
    else:
        if not all([isinstance(x, dict) for x in dict_list]):
            raise TypeError("dict_list must be a dict or a list of dicts")
    all_keys = set([])
    [all_keys.update(list(this_dict.keys())) for this_dict in
     dict_list]  # this constructs a list of all keys encountered in the list of dicts
    example_dict = dict()
    keys_remaining_to_find = all_keys
    for this_dict in dict_list:
        new_keys = list(set(keys_remaining_to_find).intersection(list(this_dict.keys())))
        if not new_keys: continue
        new_dict = {k: this_dict[k] for k in new_keys if
                    this_dict[k] or this_dict[k] == 0 or this_dict[
                        k] == True}  # keep only keys with non-empty and non-none value
        example_dict = dict(example_dict, **{k: v for k, v in list(new_dict.items())})
        keys_remaining_to_find = keys_remaining_to_find.difference(new_keys)
        if not keys_remaining_to_find: break  # if there's no more keys to be found, you can quit

    if recursive == True:
        dict_list_keys = [k for k in list(example_dict.keys()) if (k and is_dict_or_list_of_dicts(example_dict[k]))]
        for k in dict_list_keys:
            example_dict[k] = example_dict_from_dict_list(example_dict[k], recursive=True)
    return example_dict


def dict_list_key_count(dict_list):
    """
    returns a dict with all keys encoutered in the list of dicts, and values exhibiting
    how many times the key was encoutered in the dict list
    """
    all_keys = list(example_dict_from_dict_list(dict_list).keys())
    return {k: np.sum(np.array([k in d for d in dict_list])) for k in all_keys}


def dict_list_has_key_df(dict_list, index_names=None, use_0_1=False):
    """
    returns a dataframe where:
        * indices indicate what dict_list element the row corresponds to,
        * columns are all keys ever encoutered in the list of dicts, and
        * df[i,j] is True if dict i has key j
    """
    df = pd.concat([pd.Series({k: True for k in list(d.keys())}) for d in dict_list], axis=1).transpose()
    df.fillna(False, inplace=True)
    if use_0_1 == True:
        df.replace([True, False], [1, 0], inplace=True)
    if index_names:
        df.index = index_names
    return df


def dict_of_types_of_dict_values(x, recursive=False):
    if isinstance(x, list):  # if x is a list of dicts
        x = example_dict_from_dict_list(x, recursive=True)
    if not recursive:
        return {k: typeof(x[k]) for k in list(x.keys())}
    else:
        if isinstance(recursive, bool):
            next_recursive = recursive
        else:
            next_recursive = recursive - 1
        dict_of_types = dict()
        for k in list(x.keys()):
            if isinstance(x[k], dict):
                dict_of_types = dict(dict_of_types,
                                     **{k: {'dict': dict_of_types_of_dict_values(x[k], recursive=next_recursive)}})
            elif is_list_of_dicts(x[k]):
                dict_of_types = dict(dict_of_types,
                                     **{k: {
                                         'list of dict': dict_of_types_of_dict_values(x[k], recursive=next_recursive)}})
            else:
                dict_of_types = dict(dict_of_types, **{k: typeof(x[k])})
        return dict_of_types


def is_list_of_dicts(x):
    return isinstance(x, list) and len(x) > 0 and all([isinstance(xx, dict) for xx in x])


def is_dict_or_list_of_dicts(x):
    return isinstance(x, dict) or is_list_of_dicts(x)


# test_example_dict_from_dict_list()
# d1 = {'a':1, 'b':2}
# d2 = {'b':20, 'c':30}
# d12 = pdict.example_dict_from_dict_list([d1,d2])
# print d12
# print {'a':1, 'b':2, 'c':30}
# d12=={'a':1, 'b':2, 'c':30}

if __name__ == "__main__":
    t = {'a': [{
        'aa': [1, 2, 3],
        'bb': [
            {'bba': 2, 'bbb': 3},
            {'bba2': [1, 2, 3], 'bbb2': 'boo'},
            {'bba3': {'dict': 'this is a dict'}}]
    },
        {'aa2': [2, 3, 4]}],
        'b': [4, 5, 6]}
    print_dict_example_types(t, recursive=True)
