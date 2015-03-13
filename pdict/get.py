__author__ = 'thorwhalen'

import ut.util.ulist as ulist


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


def key_if_exists_else_return_none(d, key):
    DeprecationWarning('You should really call this one liner directly!!')
    return d.get(key)


def mk_fixed_coordinates_value_getter(get_key_list):
    return \
        lambda the_dict: \
            reduce(lambda x, y: x.__getitem__(y), get_key_list, the_dict)


def head(d, num_of_elements=5, start_at=0):
    """
    get the "first" few (num) elements of a dict
    """
    return {k: d[k] for k in d.keys()[start_at:min(len(d), start_at+num_of_elements)]}


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