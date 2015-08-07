__author__ = 'thor'

import numpy as np
from numpy import *

from sklearn.cluster import SpectralClustering as sk_SpectralClustering


class SpectralClustering(sk_SpectralClustering):
    def __init__(self,
                 n_clusters=8,
                 eigen_solver=None,
                 random_state=None,
                 n_init=10,
                 gamma=1.0,
                 affinity='rbf',
                 n_neighbors=10,
                 eigen_tol=0.0,
                 assign_labels='kmeans',
                 degree=3,
                 coef0=1,
                 kernel_params=None):
        super(SpectralClustering, self).__init__(
            n_clusters=n_clusters,
            eigen_solver=eigen_solver,
            random_state=random_state,
            n_init=n_init,
            gamma=gamma,
            affinity=affinity,
            n_neighbors=n_neighbors,
            eigen_tol=eigen_tol,
            assign_labels=assign_labels,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params)

    def fit(self, X, y=None):
        super(SpectralClustering, self).fit(X=X, y=y)
        self.cluster_centers_ = vstack(map(lambda i: mean(X[self.labels_==i], axis=0), sorted(unique(self.labels_))))
