from __future__ import division


def mk_inclusion_filter(include=(), key=None):
    """
    Creates a function to perform inclusion filtering (i.e. "filter-in if x is in include list")
    :param include: a list-like of elements for an inclusion filter
    :param key: None (default), a callable, or a hashable. If
        * None, the filter will be determined by: x in include
        * callable, the filter will be determined by: key(x) in include
        * hashable, the filter will be determined by: x[key] in include
    :return: a filter function (a function that returns True or False)
    """
    include = set(include)

    if key is None:
        def filter_func(x):
            return x in include
    else:
        if callable(key):
            def filter_func(x):
                return key(x) in include
        else:
            def filter_func(x):
                return x[key] in include

    return filter_func


def mk_exclusion_filter(exclude=(), key=None):
    """
    Creates a function to perform exclusion filtering (i.e. "filter-in if x is NOT in exclude list")
    :param exclude: a list-like of elements for an exclusion filter
    :param key: None (default), a callable, or a hashable. If
        * None, the filter will be determined by: x not in exclude
        * callable, the filter will be determined by: key(x) not in include
        * hashable, the filter will be determined by: x[key] not in include
    :return: a filter function (a function that returns True or False)
    """
    exclude = set(exclude)

    if key is None:
        def filter_func(x):
            return x not in exclude
    else:
        if callable(key):
            def filter_func(x):
                return key(x) not in exclude
        else:
            def filter_func(x):
                return x[key] not in exclude

    return filter_func
