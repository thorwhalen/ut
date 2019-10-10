


def mk_inclusion_exclusion_filter(include=(), exclude=(), key=None):
    """
    Convenience function to perform inclusion and exclusion filtering.
    If include is not empty and exclude is empty, it will act as mk_inclusion_filter.
    If exclude is not empty and include is empty, it will act as mk_exclusion_filter.
    If both include and exclude are not empty, it will apply mk_inclusion_filter to include - exclude.
    :param include: a list-like of elements for an inclusion filter
    :param exclude: a list-like of elements for an exclusion filter
    :param key: None (default), a callable, or a hashable. If
        * None, the filter will be determined by: x in include or x not in exclude
        * callable, the filter will be determined by: key(x) in include or key(x not in exclude
        * hashable, the filter will be determined by: x[key] in include or x[key] not in exclude
    :return: a filter function (a function that returns True or False)
    >>> # if there's no include nore exclude
    >>> filter(mk_inclusion_exclusion_filter(), [1, 2, 3, 4, 5])
    [1, 2, 3, 4, 5]
    >>> # pure inclusion demo...
    >>> filt = mk_inclusion_exclusion_filter(include=[2, 4])
    >>> filter(filt, [1, 2, 3, 4, 2, 3, 4, 3, 4])
    [2, 4, 2, 4, 4]
    >>> filt = mk_inclusion_exclusion_filter(include=[2, 4], key='val')
    >>> filter(filt, [{'name': 'four', 'val': 4}, {'name': 'three', 'val': 3}, {'name': 'two', 'val': 2}])
    [{'name': 'four', 'val': 4}, {'name': 'two', 'val': 2}]
    >>> filt = mk_inclusion_exclusion_filter(include=[2, 4], key=2)
    >>> filter(filt, [(1, 2, 3), (1, 2, 4), (2, 7, 4), (1, 2, 7)])
    [(1, 2, 4), (2, 7, 4)]
    >>> filt = mk_inclusion_exclusion_filter(include=[2, 4], key=lambda x: x[0] * x[1])
    >>> filter(filt, [(1, 2, 'is 2'), (2, 2, 'is 4'), (2, 3, 'is 6'), (1, 4, 'is 4')])
    [(1, 2, 'is 2'), (2, 2, 'is 4'), (1, 4, 'is 4')]
    >>>
    >>> # pure exclusion demo...
    >>> filt = mk_inclusion_exclusion_filter(exclude=[2, 4])
    >>> filter(filt, [1, 2, 3, 4, 2, 5, 4, 6, 4])
    [1, 3, 5, 6]
    >>> filt = mk_inclusion_exclusion_filter(exclude=[2, 4], key='val')
    >>> filter(filt, [{'name': 'four', 'val': 4}, {'name': 'three', 'val': 3}, {'name': 'two', 'val': 2}])
    [{'name': 'three', 'val': 3}]
    >>> filt = mk_inclusion_exclusion_filter(exclude=[2, 4], key=2)
    >>> filter(filt, [(1, 2, 3), (1, 2, 4), (2, 7, 4), (1, 2, 7)])
    [(1, 2, 3), (1, 2, 7)]
    >>> filt = mk_inclusion_exclusion_filter(exclude=[2, 4], key=lambda x: x[0] * x[1])
    >>> filter(filt, [(1, 2, 'is 2'), (2, 2, 'is 4'), (2, 3, 'is 6'), (1, 4, 'is 4')])
    [(2, 3, 'is 6')]
    >>>
    >>> # if include and exclude are present...
    >>> filt = mk_inclusion_exclusion_filter(include=[1, 2, 3, 4], exclude=[1, 3])
    >>> filter(filt, [1, 2, 3, 4, 2, 3, 4, 3, 4])
    [2, 4, 2, 4, 4]
    >>> filt = mk_inclusion_exclusion_filter(include=[1, 2, 3, 4], exclude=[1, 3], key='val')
    >>> filter(filt, [{'name': 'four', 'val': 4}, {'name': 'three', 'val': 3}, {'name': 'two', 'val': 2}])
    [{'name': 'four', 'val': 4}, {'name': 'two', 'val': 2}]
    >>> filt = mk_inclusion_exclusion_filter(include=[1, 2, 3, 4], exclude=[1, 3], key=2)
    >>> filter(filt, [(1, 2, 3), (1, 2, 4), (2, 7, 4), (1, 2, 7)])
    [(1, 2, 4), (2, 7, 4)]
    >>> filt = mk_inclusion_exclusion_filter(include=[1, 2, 3, 4], exclude=[1, 3], key=lambda x: x[0] * x[1])
    >>> filter(filt, [(1, 2, 'is 2'), (2, 2, 'is 4'), (2, 3, 'is 6'), (1, 4, 'is 4')])
    [(1, 2, 'is 2'), (2, 2, 'is 4'), (1, 4, 'is 4')]
    """
    if exclude:
        if not include:
            return mk_exclusion_filter(exclude=exclude, key=key)
        else:
            include = set(include).difference(exclude)
            return mk_inclusion_filter(include=include, key=key)
    else:
        if include:
            return mk_inclusion_filter(include=include, key=key)
        else:
            return lambda x: True  # transparent filter (all goes through)


def mk_inclusion_filter(include=(), key=None):
    """
    Creates a function to perform inclusion filtering (i.e. "filter-in if x is in include list")
    :param include: a list-like of elements for an inclusion filter
    :param key: None (default), a callable, or a hashable. If
        * None, the filter will be determined by: x in include
        * callable, the filter will be determined by: key(x) in include
        * hashable, the filter will be determined by: x[key] in include
    :return: a filter function (a function that returns True or False)
    >>> filt = mk_inclusion_filter(include=[2, 4])
    >>> filter(filt, [1, 2, 3, 4, 2, 3, 4, 3, 4])
    [2, 4, 2, 4, 4]
    >>> filt = mk_inclusion_filter(include=[2, 4], key='val')
    >>> filter(filt, [{'name': 'four', 'val': 4}, {'name': 'three', 'val': 3}, {'name': 'two', 'val': 2}])
    [{'name': 'four', 'val': 4}, {'name': 'two', 'val': 2}]
    >>> filt = mk_inclusion_filter(include=[2, 4], key=2)
    >>> filter(filt, [(1, 2, 3), (1, 2, 4), (2, 7, 4), (1, 2, 7)])
    [(1, 2, 4), (2, 7, 4)]
    >>> filt = mk_inclusion_filter(include=[2, 4], key=lambda x: x[0] * x[1])
    >>> filter(filt, [(1, 2, 'is 2'), (2, 2, 'is 4'), (2, 3, 'is 6'), (1, 4, 'is 4')])
    [(1, 2, 'is 2'), (2, 2, 'is 4'), (1, 4, 'is 4')]
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
    >>> filt = mk_exclusion_filter(exclude=[2, 4])
    >>> filter(filt, [1, 2, 3, 4, 2, 5, 4, 6, 4])
    [1, 3, 5, 6]
    >>> filt = mk_exclusion_filter(exclude=[2, 4], key='val')
    >>> filter(filt, [{'name': 'four', 'val': 4}, {'name': 'three', 'val': 3}, {'name': 'two', 'val': 2}])
    [{'name': 'three', 'val': 3}]
    >>> filt = mk_exclusion_filter(exclude=[2, 4], key=2)
    >>> filter(filt, [(1, 2, 3), (1, 2, 4), (2, 7, 4), (1, 2, 7)])
    [(1, 2, 3), (1, 2, 7)]
    >>> filt = mk_exclusion_filter(exclude=[2, 4], key=lambda x: x[0] * x[1])
    >>> filter(filt, [(1, 2, 'is 2'), (2, 2, 'is 4'), (2, 3, 'is 6'), (1, 4, 'is 4')])
    [(2, 3, 'is 6')]
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
