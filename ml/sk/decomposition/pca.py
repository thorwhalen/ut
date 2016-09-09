from __future__ import division

from sklearn.decomposition.pca import PCA
from ut.ml.sk.utils.validation import weighted_data
from numpy import reshape

__author__ = 'thor'


class WeightedPCA(PCA):
    """
    Weighted version of sklearn.decomposition.pca.PCA
>>> from sklearn.decomposition.pca import PCA
>>> from ut.ml.sk.decomposition.pca import WeightedPCA
>>> from sklearn.datasets import make_blobs
>>> from ut.ml.sk.preprocessing import WeightedStandardScaler
>>> from numpy import ones, vstack, hstack, random
>>> from ut.ml.sk.utils.validation import compare_model_attributes, repeat_rows
>>>
>>> model_1 = PCA()
>>> model_2 = WeightedPCA()
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
>>> w = random.randint(1, 5, len(X))
>>> compare_model_attributes(model_1.fit(repeat_rows(X, w)), model_2.fit((X, w)))
all fitted attributes were close
    """
    def _fit(self, X):
        X, w = weighted_data(X)
        X = (X.T * reshape(w, (1, len(X)))).T
        return super(self.__class__, self)._fit(X)
