__author__ = 'thor'

import collections
from collections import OrderedDict


def sort_by_keys(d, reverse=False):
    return OrderedDict(sorted(list(d.items()), key=lambda t: t[0], reverse=reverse))


def sort_by_value(d, reverse=False):
    return OrderedDict(sorted(list(d.items()), key=lambda t: t[1], reverse=reverse))


def sort_by_function(d, fun, reverse=False):
    return OrderedDict(sorted(list(d.items()), key=fun, reverse=reverse))


def ordered_counter(x, sort_by=None, reverse=False):
    d = OrderedDict(collections.Counter(x))
    if sort_by is not None:
        if sort_by == 'count':
            d = sort_by_keys(d, reverse=reverse)
        elif sort_by == 'value':
            d = sort_by_value(d, reverse=reverse)
        else:
            d = sort_by_function(d, fun=sort_by, reverse=reverse)
    return d
