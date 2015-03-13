__author__ = 'thorwhalen'

import numpy as np

def contains(A, B):
    return len({x for x in B if x in A}) == len(set(B))


def unique(X):
    type(X)(np.unique(X))


def union(A, B):
    return type(A)(set(A)|set(B))


def intersect(A, B):
    return type(A)(set(B)&set(A))


def setdiff(A, B):
    return type(A)(set(A)-set(B))


def ismember(a, b):
    if isinstance(a, list):
        return not not np.intersect1d(a, b)
    else:
        return not not np.intersect1d([a], b)
