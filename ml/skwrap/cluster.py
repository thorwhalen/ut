__author__ = 'thor'

import numpy as np
from numpy import *

from sklearn.cluster import SpectralClustering as sk_SpectralClustering


class SpectralClustering(sk_SpectralClustering):
    def __init__(self, *args, **kwargs):
        super(SpectralClustering, self).__init__(*args, **kwargs)

    def fit(self, X, y=None):
        super(SpectralClustering, self).fit(X=X, y=y)
        self.cluster_centers_ = vstack(map(lambda i: mean(X[self.labels_==i], axis=0), sorted(unique(self.labels_))))
