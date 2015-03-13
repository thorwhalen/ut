__author__ = 'thor'

from numpy import *
import pandas as pd


def set_containment_matrix(family_of_sets, family_of_sets_2=None):
    """
    Computes the containment incidence matrix of two families of sets A and B, where A and B are specified by
    incidence matrices where rows index sets and columns index elements (so they must have the same number of cols).

    The function returns a boolean matrix M of dimensions nrows(A) x nrows(B) where M(i,j)==True if and only if
    ith set of A contains (or is equal to) jth set of B.

    If the second family of sets is not given, the function computes the containment matrix of the first input on
    itself.

    See Also:
        family_of_sets_to_bitmap and bitmap_to_family_of_sets to transform family of sets specification
        (useful to transform) input and output
    Example:
        >> import itertools
        >> t = array([x for x in itertools.product(*([[0, 1]] * 3))]).astype(int32)
        >> t
        =  array([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 1],
                   [1, 0, 0],
                   [1, 0, 1],
                   [1, 1, 0],
                   [1, 1, 1]], dtype=int32)
        >> bitmap_to_family_of_sets(set_containment_matrix(t), range(len(t)))
        =   [array([0]),
             array([0, 1]),
             array([0, 2]),
             array([0, 1, 2, 3]),
             array([0, 4]),
             array([0, 1, 4, 5]),
             array([0, 2, 4, 6]),
             array([0, 1, 2, 3, 4, 5, 6, 7])]

    """
    if family_of_sets_2 is None:
        family_of_sets_2 = family_of_sets
    x = matrix((~family_of_sets.astype(bool)).astype(int))
    xx = matrix((family_of_sets_2.astype(bool)).astype(int)).T
    return squeeze(asarray(~(((x * xx)).astype(bool))))


def family_of_sets_to_bitmap(family_of_sets, output='df'):
    """
    Takes set_list, a family of sets, and returns a dataframe of bitmaps (if output=='df')
    or a bitmap matrix and array of element names,
    whose columns index set elements and rows index sets.
    See Also:
        bitmap_to_family_of_sets(bitmap, set_labels) (reverse operation)
    """
    df = pd.DataFrame([{element: 1 for element in s} for s in family_of_sets], index=family_of_sets).fillna(0)
    if output == 'df':
        return df
    else:
        return df.as_matrix(), array(df.columns)


def bitmap_to_family_of_sets(bitmap, set_labels=None):
    """
    Takes a bitmap specification of a family of sets and returns an list of arrays specification.
    Input:
        bitmap: a dataframe whose rows index sets and columns index set labels
        bitmap, set_labels: here bitmap is a (sets x set_element_idx) matrix and set_labels are the set elements
    See Also:
        family_of_sets_to_bitmap(family_of_sets) (reverse operation)
    """
    if isinstance(bitmap, pd.DataFrame):
        if set_labels is None:
            set_labels = array(bitmap.columns)
        bitmap = bitmap.as_matrix()
    else:
        if set_labels is None:
            set_labels = arange(shape(bitmap)[1])
    assert shape(bitmap)[1] == len(set_labels), "number of set labels must equal the number of elements (num of cols)"
    return [set_labels[lidx] for lidx in bitmap]


class SetFamily(object):
    def __init__(self, set_family, element_labels=None):
        if isinstance(set_family, pd.DataFrame):
            element_labels = array(set_family.columns)
            self.set_family = set_family.as_matrix() == 1
        elif isinstance(set_family, ndarray) and len(shape(set_family)) == 2:
            self.set_family = set_family
        else:
            self.set_family, element_labels = family_of_sets_to_bitmap()
        self.n_sets = shape(self.set_family)[0]
        self.n_set_elements = shape(self.set_family)[1]
        self.set_cardinalities = sum(self.set_family, 1).reshape((self.n_sets, 1))
        if element_labels is None:
            self.element_labels = arange(self.n_set_elements)
        else:
            self.element_labels = element_labels
