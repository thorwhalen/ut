from __future__ import division

from sklearn.decomposition.pca import PCA, _infer_dimension_
from ut.ml.sk.utils.validation import weighted_data
from wpca import WPCA
from numpy import reshape, tile

from numpy import reshape, sqrt, average

import numpy as np
from scipy import linalg
# from scipy.special import gammaln

from sklearn.utils.validation import as_float_array
from sklearn.utils.validation import check_array
# from sklearn.utils.extmath import fast_dot, fast_logdet, randomized_svd
# from sklearn.utils.validation import check_is_fitted

__author__ = 'thor'

#
# class WeightedPCA(WPCA):
#     """
#     Weighted version of sklearn.decomposition.pca.PCA
# >>> from sklearn.decomposition.pca import PCA
# >>> from ut.ml.sk.decomposition.pca import WeightedPCA
# >>> from sklearn.datasets import make_blobs
# >>> from ut.ml.sk.preprocessing import WeightedStandardScaler
# >>> from numpy import ones, vstack, hstack, random
# >>> from ut.ml.sk.utils.validation import compare_model_attributes, repeat_rows
# >>>
# >>> model_1 = PCA()
# >>> model_2 = WeightedPCA()
# >>>
# >>> X, y = make_blobs(100, 5, 4)
# >>> w = ones(len(X))
# >>> compare_model_attributes(model_1.fit(X), model_2.fit(X))
# all fitted attributes were close
# >>>
# >>> X, y = make_blobs(100, 5, 4)
# >>> w = ones(len(X))
# >>>
# >>> XX = vstack((X, X))
# >>> wX = (X, 2 * ones(len(X)))
# >>> compare_model_attributes(model_1.fit(XX), model_2.fit(wX))
# all fitted attributes were close
# >>>
# >>> X, y = make_blobs(100, 5, 4)
# >>> w = ones(len(X))
# >>>
# >>> XX = vstack((X, X[-2:, :], X[-1, :]))
# >>> wX = (X, hstack((ones(len(X)-2), [2, 3])))
# >>> compare_model_attributes(model_1.fit(XX), model_2.fit(wX))
# all fitted attributes were close
# >>>
# >>> w = random.randint(1, 5, len(X))
# >>> compare_model_attributes(model_1.fit(repeat_rows(X, w)), model_2.fit((X, w)))
# all fitted attributes were close
#     """
#     def fit(self, X, y=None):
#         X, w = weighted_data(X)
#         return super(self.__class__, self).fit(X, weights=tile(reshape(w, (len(w), 1)), X.shape[1]))
#

class MyWeightedPCA(PCA):
    def _fit(self, X):
        X, w = weighted_data(X)

        X = check_array(X)
        n_samples, n_features = X.shape
        n_samples_weighted = sum(w)
        X = as_float_array(X, copy=self.copy)
        # Center data
        # self.mean_ = average(X, axis=0, weights=w)
        # X -= self.mean_
        U, S, V = linalg.svd((X.T * reshape(sqrt(w), (1, len(X)))).T, full_matrices=True)
        explained_variance_ = (S ** 2) / n_samples_weighted
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())

        components_ = V

        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")

            n_components = _infer_dimension_(explained_variance_,
                                             n_samples, n_features)
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = np.sum(ratio_cumsum < n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples_weighted # n_samples

        self.components_ = components_[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return (U, S, V)






        #
        # # Center data
        # self.mean_ = average(X, axis=0, weights=w)
        # X -= self.mean_
        #
        # X = (X.T * reshape(sqrt(w), (1, len(X)))).T
        # return super(self.__class__, self)._fit(X)
