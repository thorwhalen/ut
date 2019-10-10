"""
Neighborhood Components Analysis (NCA)
Ported to Python from https://github.com/vomjom/nca
"""

import numpy as np

from numpy.linalg import inv, cholesky
from  matplotlib.pyplot import fill_betweenx


class BaseMetricLearner(object):
    def __init__(self):
        raise NotImplementedError('BaseMetricLearner should not be instantiated')

    def metric(self):
        L = self.transformer()
        return L.T.dot(L)

    def transformer(self):
        return inv(cholesky(self.metric()))

    def transform(self, X=None):
        if X is None:
            X = self.X
        L = self.transformer()
        return L.dot(X.T).T


# BaseEstimator, TransformerMixin

class NCA(BaseMetricLearner):
    def __init__(self, max_iter=100, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.A = None

    def transformer(self):
        return self.A

    def fit(self, X, y):
        """
        X: data matrix, (n x d)
        y: scalar y, (n)
        """
        n, d = X.shape
        # Initialize A to a scaling matrix
        A = np.zeros((d, d))
        np.fill_diagonal(A, 1. / (X.max(axis=0) - X.min(axis=0)))

        # Run NCA
        dX = X[:, None] - X[None]  # shape (n, n, d)
        tmp = np.einsum('...i,...j->...ij', dX, dX)  # shape (n, n, d, d)
        masks = y[:, None] == y[None]
        for it in range(self.max_iter):
            for i, label in enumerate(y):
                mask = masks[i]
                Ax = A.dot(X.T).T  # shape (n, d)

                softmax = np.exp(-((Ax[i] - Ax) ** 2).sum(axis=1))  # shape (n)
                softmax[i] = 0
                softmax /= softmax.sum()

                t = softmax[:, None, None] * tmp[i]  # shape (n, d, d)
                d = softmax[mask].sum() * t.sum(axis=0) - t[mask].sum(axis=0)
                A += self.learning_rate * A.dot(d)

        self.X = X
        self.A = A
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
        # return self.transform(X)








