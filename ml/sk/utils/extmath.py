from __future__ import division

from numpy import ones, reshape, sum

from ut.ml.sk.utils.validation import weighted_data

__author__ = 'thor'


def _weighted_incremental_mean_and_var(X, last_mean=.0, last_variance=None,
                              last_sample_count=.0):


    """Calculate mean update and a Youngs and Cramer variance update with weighted data.

    This is a based on the sklearn function.


    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0.
    In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : 2-tuple (X, w) where X is array-like, shape (n_samples, n_features) and weight is a (n_samples,) array
    indicating the weight of each row of X
        Data to use for variance update


    last_mean : array-like, shape: (n_features,)

    last_variance : array-like, shape: (n_features,)

    last_sample_count : int

    Returns
    -------
    updated_mean : array, shape (n_features,)

    updated_variance : array, shape (n_features,)
        If None, only mean is computed

    updated_sample_count : int

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats

    X, w = weighted_data(X)

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

    wX = (X.T * w.T).T

    new_sample_count = sum(w)
    updated_sample_count = last_sample_count + new_sample_count

    last_sum = last_mean * last_sample_count

    # print last_sum, new_sum, updated_sample_count
    updated_mean = (last_sum + wX.sum(axis=0)) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        if last_sample_count == 0:  # Avoid division by 0
            new_unnormalized_variance = X.var(axis=0) * new_sample_count
            updated_variance = new_unnormalized_variance / updated_sample_count
        else:
            last_mean_of_squares = last_variance + last_mean ** 2  # because E(X^2) = Var(X) + E(X)^2
            last_sum_of_squares = last_mean_of_squares * last_sample_count
            updated_sum_of_squares = last_sum_of_squares + sum(wX ** 2, axis=0)
            updated_mean_of_squares = updated_sum_of_squares / updated_sample_count
            updated_variance = updated_mean_of_squares - updated_mean ** 2

    return updated_mean, updated_variance, updated_sample_count
