__author__ = 'thor'


import collections


def duplicated_values(arr):
    return [x for x, y in collections.Counter(arr).items() if y > 1]