

__author__ = 'thor'

from itertools import islice


def head(rdd, k=5):
    if k >= 0:
        return [x for x in islice(rdd.toLocalIterator(), k)]
    else:
        return [x for x in islice(rdd.toLocalIterator(), rdd.count() + k, rdd.count())]
