from __future__ import division

__author__ = 'thor'

from sklearn.decomposition import IncrementalPCA as IncrementalPCA_sk
from numpy import vstack, ndarray


class IncrementalPCA(IncrementalPCA_sk):
    """
    sklearn's IncrementalPCA doesn't work if you start off with less data than you want components

    This wrapper is to avoid this. It will accumulate the data fed by partial_fit until there's enough data to start
    doing the normal (sklearn's) partial_fit.
    """
    def __init__(self, n_components=None, whiten=False, copy=True, batch_size=None, X_cumul=None):
        super(self.__class__, self).__init__(n_components=n_components, whiten=whiten,
                                             copy=copy, batch_size=batch_size)
        self.X_cumul = X_cumul
        self.fitted_ = False
        if n_components is not None and batch_size is not None:
            self.batch_threshold = min(self.n_components, self.batch_size)

    def partial_fit(self, X, y=None):
        if self.n_components is None:
            self.n_components = X.shape[1]
        if self.batch_size is None:
            self.batch_threshold = self.n_components
        else:
            self.batch_threshold = min(self.n_components, self.batch_size)

        if self.X_cumul is not None:
            self.X_cumul = vstack((self.X_cumul, X))
        else:
            self.X_cumul = X

        if self.X_cumul.shape[0] >= self.batch_threshold:
            # you have enough, you can do the first partial_fit now!
            self.flush_x_cumul()
        else:
            self.fitted_ = False

        return self

    def flush_x_cumul(self):
        if self.X_cumul.shape[0] > 0:
            super(self.__class__, self).partial_fit(self.X_cumul)
            self.X_cumul = ndarray(shape=(0, self.X_cumul.shape[1]))
            self.fitted_ = True

    def transform(self, X):
        self.flush_x_cumul()
        return super(self.__class__, self).transform(X)

