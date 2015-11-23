from __future__ import division

__author__ = 'thor'
__doc__ = """
Functions to do spectral clustering as I'd like to.
That is:
    * Wrapping existing spectral functionality in an API that is more intuitive to me (both input and output)
"""

import spectral
import pandas as pd

from ut.ml.cluster.util import order_clus_idx_and_centers_by_decreasing_frequencies


def kmeans(X, max_clusters=10, max_iterations=20, distance='euclidean', **kwargs):

    # input processing ########################################################################
    X, distance = _how_spectral_library_expects_data_and_distance(X, distance)

    # do the clustering (call spectral.kmeans)
    clus_idx, clus_centers = spectral.kmeans(X,
                                             nclusters=max_clusters,
                                             max_iterations=max_iterations,
                                             distance=distance,
                                             **kwargs)

    return _how_I_like_clus_idx_and_clus_centers(clus_idx, clus_centers)


def _how_spectral_library_expects_data_and_distance(X, distance):
    # X, the data
    if isinstance(X, pd.DataFrame):
        X = X.as_matrix()
    X = X.reshape([X.shape[0], 1, X.shape[1]])

    # distance
    if distance not in [spectral.clustering.L1, spectral.clustering.L2]:
        if callable(distance):
            distance = distance.__name__
        if isinstance(distance, basestring):
            if distance.lower() in ['euclidean', 'l2']:
                distance = spectral.clustering.L2
            elif distance.lower() in ['cityblock', 'manhattan', 'l1']:
                distance = spectral.clustering.L1
            else:
                raise ValueError("Unknown distance specification: {}".format(distance))
        else:
            raise ValueError("distance should be a string or a callable")

    return X, distance


def _how_I_like_clus_idx_and_clus_centers(clus_idx, clus_centers):
    return order_clus_idx_and_centers_by_decreasing_frequencies(clus_idx[:, 0], clus_centers)





