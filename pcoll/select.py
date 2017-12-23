from __future__ import division

inf = float('inf')


def sublist_that_contains_segment(sorted_vals, from_val=None, to_val=None, key=None):
    """
    Returns the sublist of a sorted iterator that tightly contains both from_val and to_val.
     That is, it is the smallest sublist containing the [from_val, to_val] interval.
    :param sorted_vals: A sorted iterator
    :param from_val: The smallest value to contain
    :param to_val: The largest value to contain
    :param key: How to get the vals (to compare with) from the sorted_vals iterator elements.
        * None: Will just take the element itself
        * callable: Will take key(element) as the val
        * else will take element[key] as the val (hoping that key is a hashable, and exists as a key to element)
    :return: A list of the elements of sorted_vals.
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
