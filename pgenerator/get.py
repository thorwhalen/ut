__author__ = 'thorwhalen'

import glob
import itertools


def files_matching(file_path_pattern):
    """
    returns a generator for files matching the file_path_pattern
    WARNING: The pattern seems to be a unix style one (i.e. "normal" regex doesn't work)
    """
    return glob.iglob(file_path_pattern)


def lines_matching(lines, searchtext):
    for line in lines:
        if searchtext in line: yield line


def last_element(g):
    """
    DEPRECATED! Use pgenerator.util instead
    """
    DeprecationWarning("Use pgenerator.util instead")
    if hasattr(g,'__reversed__'):
        last = next(reversed(g))
    else:
        for last in g:
            pass
    return last


def dict_vals_product(dict_of_arrays):
    return (dict(zip(dict_of_arrays, x)) for x in itertools.product(*iter(dict_of_arrays.values())))


def chunks(n, iterable):
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))
