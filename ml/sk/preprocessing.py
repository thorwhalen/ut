from __future__ import division

__author__ = 'thor'

import numpy as np
from ut.ml.sk.utils.validation import weighted_data
from numpy import allclose

from statsmodels.stats.weightstats import DescrStatsW
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import (DEPRECATION_MSG_1D, FLOAT_DTYPES, check_array, warnings, sparse,
                                        _incremental_mean_and_var, mean_variance_axis, incr_mean_variance_axis,
                                        _handle_zeros_in_scale)


class WeightedStandardScaler(StandardScaler):
    """
    A version of sklearn.preprocessing.StandardScaler that works with weighted data.

>>> from sklearn.datasets import make_blobs
>>> from ut.ml.sk.preprocessing import WeightedStandardScaler
>>> from sklearn.preprocessing import StandardScaler
>>> from numpy import ones, vstack, hstack, random
>>> from ut.ml.sk.utils.validation import compare_model_attributes, repeat_rows
>>>
>>> model_1 = StandardScaler()
>>> model_2 = WeightedStandardScaler()
>>>
>>> X, y = make_blobs(100, 5, 4)
>>> w = ones(len(X))
>>> compare_model_attributes(model_1.fit(X), model_2.fit(X))
all fitted attributes were close
>>>
>>> X, y = make_blobs(100, 5, 4)
>>> w = ones(len(X))
>>>
>>> XX = vstack((X, X))
>>> wX = (X, 2 * ones(len(X)))
>>> compare_model_attributes(model_1.fit(XX), model_2.fit(wX))
all fitted attributes were close
>>>
>>> X, y = make_blobs(100, 5, 4)
>>> w = ones(len(X))
>>>
>>> XX = vstack((X, X[-2:, :], X[-1, :]))
>>> wX = (X, hstack((ones(len(X)-2), [2, 3])))
>>> compare_model_attributes(model_1.fit(XX), model_2.fit(wX))
all fitted attributes were close
>>>
>>> assert allclose(model_1.transform(X), model_2.transform(X)), "transformation of X not close"
>>>
>>> w = random.randint(1, 5, len(X))
>>> compare_model_attributes(model_1.fit(repeat_rows(X, w)), model_2.fit((X, w)))
all fitted attributes were close
>>>
    """
    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y: Passthrough for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()

        X, w = weighted_data(X)

        weighted_stats = DescrStatsW(X, weights=w, ddof=0)

        self.mean_ = weighted_stats.mean  # weighted mean of data (equivalent to np.average(array, weights=weights))
        self.var_ = weighted_stats.var  # variance with default degrees of freedom correction
        self.n_samples_seen_ = sum(w)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def partial_fit(self, X, y=None):
        """Online computation of mean and std on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y: Passthrough for ``Pipeline`` compatibility.
        """
        raise NotImplementedError("Partial fit for WeightedStandardScaler not yet implemented")
        #
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 ensure_2d=False, warn_on_dtype=True,
        #                 estimator=self, dtype=FLOAT_DTYPES)
        #
        # if X.ndim == 1:
        #     warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)
        #
        # # Even in the case of `with_mean=False`, we update the mean anyway
        # # This is needed for the incremental computation of the var
        # # See incr_mean_variance_axis and _incremental_mean_variance_axis
        #
        # if sparse.issparse(X):
        #     if self.with_mean:
        #         raise ValueError(
        #             "Cannot center sparse matrices: pass `with_mean=False` "
        #             "instead. See docstring for motivation and alternatives.")
        #     if self.with_std:
        #         # First pass
        #         if not hasattr(self, 'n_samples_seen_'):
        #             self.mean_, self.var_ = mean_variance_axis(X, axis=0)
        #             self.n_samples_seen_ = X.shape[0]
        #         # Next passes
        #         else:
        #             self.mean_, self.var_, self.n_samples_seen_ = \
        #                 incr_mean_variance_axis(X, axis=0,
        #                                         last_mean=self.mean_,
        #                                         last_var=self.var_,
        #                                         last_n=self.n_samples_seen_)
        #     else:
        #         self.mean_ = None
        #         self.var_ = None
        # else:
        #     # First pass
        #     if not hasattr(self, 'n_samples_seen_'):
        #         self.mean_ = .0
        #         self.n_samples_seen_ = 0
        #         if self.with_std:
        #             self.var_ = .0
        #         else:
        #             self.var_ = None
        #
        #     self.mean_, self.var_, self.n_samples_seen_ = \
        #         _incremental_mean_and_var(X, self.mean_, self.var_,
        #                                   self.n_samples_seen_)
        #
        # if self.with_std:
        #     self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        # else:
        #     self.scale_ = None
        #
        # return self



def compare_unweighted_to_weighted(X, wX):
    ss = StandardScaler()
    wss = WeightedStandardScaler()

    ss.fit(X)
    wss.fit(wX)
    try:
        assert allclose(ss.mean_, wss.mean_), 'mean_ not close'
        assert allclose(ss.var_, wss.var_), 'var_ not close'
        assert allclose(ss.scale_, wss.scale_), 'scale_ not close'
        print("all okay")
    except AssertionError as e:
        print(e)

    return ss, wss
