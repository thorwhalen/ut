__author__ = 'thor'


def last_element(g):
    """
    returns last element of a generator
    """
    if hasattr(g, '__reversed__'):
        last = next(reversed(g))
    else:
        for last in g:
            pass
    return last


