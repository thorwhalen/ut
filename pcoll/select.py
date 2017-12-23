from __future__ import division

inf = float('inf')


def sublist_that_contains_segment(sorted_vals, from_val=-inf, to_val=inf, key=None):
    """
    Returns the sublist of a sorted iterator that tightly contains both from_val and to_val.
     In some sense, it is the smallest sublist containing the [from_val, to_val] interval.
     We say "in some sense" because the function will range from the largest value strictly smaller than from_val to
     the smallest value strictly greater than to_val.
     For example, if sorted_vals = [0, 5, 10, 15, 20, 25, 30], from_val=10, and to_val=20, the function will return
        [5, 10, 15, 20, 25]
     as opposed to
        [10, 15, 20, 25]
    This is useful when the sorted_vals are actually taken from a larger sorted list and you want to make sure that
    the interval returned contains ALL vals that are greater OR EQUAL to from_val and smaller OR EQUAL to to_val.
    We call this the "superset-safe" property.
    :param sorted_vals: A sorted iterator
    :param from_val: The smallest value to contain. If None will take from_val=-inf
    :param to_val: The largest value to contain. If None will take from_val=inf
    :param key: How to get the vals (to compare with) from the sorted_vals iterator elements.
        * None: Will just take the element itself
        * callable: Will take key(element) as the val
        * else will take element[key] as the val (hoping that key is a hashable, and exists as a key to element)
    :return: A list of the elements of sorted_vals.
    >>> sorted_vals = [0, 5, 10, 15, 20, 25, 30]
    >>> sublist_that_contains_segment(sorted_vals, 11, 21)
    [10, 15, 20, 25]
    >>> # test the superset-safe property (see description above)
    >>> sublist_that_contains_segment(sorted_vals, 10, 20)
    [5, 10, 15, 20, 25]
    >>> # test specifying only from_val
    >>> sublist_that_contains_segment(sorted_vals, from_val=11)
    [10, 15, 20, 25, 30]
    >>> # test specifying only to_val
    >>> sublist_that_contains_segment(sorted_vals, to_val=21)
    [0, 5, 10, 15, 20, 25]
    >>> # test a functional key
    >>> sorted_vals = [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 3, 'b': 4}, {'a': 4, 'b': 5}, {'a': 5, 'b': 6}]
    >>> sublist_that_contains_segment(sorted_vals, 7, 12, key=lambda x: x['a'] * x['b'])
    [{'a': 2, 'b': 3}, {'a': 3, 'b': 4}, {'a': 4, 'b': 5}]
    >>> # test a hashable key
    >>> sorted_vals = [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 3, 'b': 4}, {'a': 4, 'b': 5}, {'a': 5, 'b': 6}]
    >>> sublist_that_contains_segment(sorted_vals, 2.1, 3, key='a')
    [{'a': 2, 'b': 3}, {'a': 3, 'b': 4}, {'a': 4, 'b': 5}]
    """
    sublist = list()
    if from_val is None:
        from_val = -inf
    if to_val is None:
        to_val = inf
    if key is None:
        x_to_val = lambda x: x
    elif callable(key):
        x_to_val = key
    else:
        x_to_val = lambda x: x[key]

    previous_val = -inf
    for x in sorted_vals:
        val = x_to_val(x)
        if val < previous_val:
            raise AssertionError("sorted_ts is not sorted")
        if val < from_val:
            sublist = [x]
        else:
            sublist.append(x)
        if val > to_val:
            break
        previous_val = val

    return sublist
