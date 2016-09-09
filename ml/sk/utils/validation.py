from __future__ import division

from numpy import reshape, ones, allclose, tile, random, vstack

__author__ = 'thor'


def weighted_data(X):
    """
    Takes an object X and returns a tuple X, w where X is a (n_samples, n_features) array and w is a (nsamples,) array
    corresponding to weights of the rows of X.

    """
    if isinstance(X, tuple):
        if len(X) == 1:
            X = X[0]
            if len(X.shape) == 1:
                X = reshape(X, (len(X), 1))
            elif len(X.shape) > 2:
                raise ValueError("data must be a matrix with no more than 2 dimensions")
            w = ones(X.shape[0])
        elif len(X) > 2:
            raise ValueError("X must be a 2-tuple of data (matrix) and weights")
        else:
            X, w = X
    else:
        w = ones(X.shape[0])

    return X, w


def compare_model_attributes(model_1, model_2, rtol=1e-05, atol=1e-08, equal_nan=False):

    try:
        for attr in model_1.__dict__.keys():
            if attr.endswith('_') and attr in model_2.__dict__:
                assert allclose(getattr(model_1, attr), getattr(model_2, attr),
                                rtol=rtol, atol=atol, equal_nan=equal_nan), \
                    '{} of {} and {} not close'.format(attr, model_1.__class__, model_2.__class__)
        print("all fitted attributes where close")
    except AssertionError as e:
        print(e)


def repeat_rows(X, row_repetition=None):
    if row_repetition is None:
        row_repetition = random.rand(1, 5, len(X))
    return vstack(map(lambda row_and_weight: tile(row_and_weight[0], (row_and_weight[1], 1)),
                      zip(X[:5], row_repetition)))