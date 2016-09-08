from __future__ import division

from numpy import reshape, ones

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