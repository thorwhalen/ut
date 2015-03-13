__author__ = 'thorwhalen'

from ut.pcoll.ordered_set import OrderedSet


def unique(X):
    seen = set()
    seen_add = seen.add
    return type(X)([x for x in X if x not in seen and not seen_add(x)])


def union(A, B):
    C = OrderedSet(B) | OrderedSet(A)
    try:
        return type(A)(C)
    except TypeError:
        return list(C)


def intersect(A, B):
    C = OrderedSet(B) & OrderedSet(A)
    try:
        return type(A)(C)
    except TypeError:
        return list(C)


def setdiff(A, B):
    # C = OrderedSet(union(B, A)) - OrderedSet(intersect(B, A)) # no, that's the symmetric difference!
    C = OrderedSet(A) - OrderedSet(intersect(B, A))
    try:
        return type(A)(C)
    except TypeError:
        return list(C)


def reorder_as(A, B):
    """
    reorders A so as to respect the order in B.
    Only the elements of A that are also in B will be reordered (and placed in front),
    those that are not will be put at the end of the returned iterable, in their original order
    """
    C = intersect(B, A) + setdiff(A, B)
    try:
        return type(A)(C)
    except TypeError:
        return list(C)
