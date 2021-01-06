"""Utils to work with lists"""

__author__ = 'thorwhalen'

import numpy as np
import ut.util.var as util_var

from heapq import heappushpop, heappush
from numpy import array, argsort


class KeepMaxK(list):
    def __init__(self, k):
        super(self.__class__, self).__init__()
        self.k = k

    def push(self, item):
        if len(self) >= self.k:
            heappushpop(self, item)
        else:
            heappush(self, item)


class KeepMaxUnikK(object):
    def __init__(self, k):
        self.min_val_items = KeepMaxK(k)
        self.item_set = set()

    def push(self, item, val):
        if item not in self.item_set:
            self.item_set.add(item)
            self.min_val_items.push((val, item))

    def items_sorted(self):
        dists, items = list(zip(*self.min_val_items))
        return array(items)[argsort(dists)]


class KeepMinK(list):
    """
    Does what KeepMaxK does, but with min.
    NOTE: Only works with items that are pairs. This is because handling the more general case makes the push two
    times slower (overhead due to handling various cases).
    If you try to push items that are not list-like, it will raise a TypeError.
    If you push items that have only one element, it will raise an IndexError.
    If you push items that have more than 2 elements, only the first two will be taken into account.
    """

    def __init__(self, k):
        super(self.__class__, self).__init__()
        self.k = k

    def push(self, item):
        # try:
        #     item = [-item[0]] + list(item[1:])
        # except TypeError:
        #     item = -item

        if len(self) >= self.k:
            heappushpop(self, (-item[0], item[1]))
        else:
            heappush(self, (-item[0], item[1]))

    def get_list(self):
        return [(-item[0], item[1]) for item in self]


def first_non_zero(arr):
    """
    Return the index of the first element that is not zero, None, nan, or False
    (basically, what ever makes "if element" True)
    The other way to do this would be where(arr)[0][0], but in some cases (large array, few non zeros))
    this function is quicker.
    Note: If no element is "non zero", the function returns None
    """
    for i, a in enumerate(arr):
        if a:
            return i
    return None


def first_true_cond(cond, arr):
    """
    Return the index of the first element such that cond(element) returns True.
    The other way to do this would be where(map(cond, arr))[0][0], of where(cond(arr))[0][0] if cond is vectorized,
    but in some cases (large array, few cond(element) == True)) this function is quicker.
    Note: If no element satisfies the condition, the function returns None
    """
    for i, a in enumerate(arr):
        if cond(a):
            return i
    return None


def get_first_item_contained_in_intersection_of(find_first_item_of, in_iterable, default=None):
    for item in in_iterable:
        if item in find_first_item_of:
            return item
    return default


def ismember_lidx(A, B):
    # returns an A-length bitmap specifying what elements of A are in B
    return [i in B for i in A]


# def ismember(a, b):
#     # tf = np.in1d(a,b) # for newer versions of numpy
#     tf = np.array([i in b for i in a])
#     u = np.unique(a[tf])
#     index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
#     return tf, index

def sort_as(sort_x, as_y, **sorted_kwargs):
    return [x for (y, x) in sorted(zip(as_y, sort_x), **sorted_kwargs)]


def all_true(x):
    """
    returns true if all entries of x are true
    """
    return np.all(x)
    # return len([xx for xx in x if xx==True]) == len(x)


def any_true(x):
    """
    returns true if any entries of x are true
    """
    return np.any(x)
    # return len([xx for xx in x if xx==True]) != 0


def to_str(x, sep=","):
    return sep.join(ascertain_list(x))


def ascertain_list(x):
    """
    ascertain_list(x) blah blah returns [x] if x is not already a list, and x itself if it's already a list
    Use: This is useful when a function expects a list, but you want to also input a single element without putting this
    this element in a list
    """
    if not isinstance(x, list):
        ## The "all but where it's a problem approach"
        if util_var.is_an_iter(x) and not isinstance(x, dict):
            x = list(x)
        else:
            x = [x]
            ## The (less safe) "just force what you want to do differently in those cases only" approach
            # if isinstance(x, np.ndarray):
            #     x = list(x)
            # else:
            #     x = [x]
    return x
