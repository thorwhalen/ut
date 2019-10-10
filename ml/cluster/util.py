

__author__ = 'thor'

from collections import Counter
from numpy import array, argsort, ones
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.spatial.distance import cdist
from ut.ml.cluster.w_kmeans import KMeansWeighted
# from sound_sketch.sb.WeightedMiniBatchKMeans import MiniBatchKMeans as KMeansWeighted
import pandas as pd


def reduce_weighted_pts(X, weights=None, reduce_to_npts=None):
    if reduce_to_npts is not None and reduce_to_npts < len(X):
        if weights is None:
            km = KMeans(n_clusters=reduce_to_npts).fit(X)
            clusters = km.predict(X)
            cluster_weights = pd.Series(Counter(clusters)).sort_index()
            return km.cluster_centers_, array(cluster_weights)
        else:
            wkm = KMeansWeighted(n_clusters=reduce_to_npts).fit(X, weights)
            clusters = wkm.predict(X)
            cluster_weights = pd.DataFrame({'clusters': clusters, 'weights': weights})
            cluster_weights = cluster_weights.groupby('clusters').sum().sort_index()
            return wkm.cluster_centers_, array(cluster_weights['weights'])
    else:
        return X, weights


# def reduce_weighted_pts(X, weights=None, reduce_to_npts=None, search_n_neighbors=4, metric='euclidean'):
#     raise NotImplementedError("not finished implementing this")
#     if weights is None:
#         weights = ones(len(X))
#     if reduce_to_npts is not None and reduce_to_npts < len(X):
#         knn = NearestNeighbors(n_neighbors=search_n_neighbors, metric=metric)
#         g = sparse.triu(knn.fit(X).kneighbors_graph(X, mode='distance'), k=1).tocsr()
#         # g = knn.fit(X).kneighbors_graph(X, mode='distance')
#
#         idx1, idx2 = g.nonzero()  # get the nonzero pairs
#
#         weighted_distance = \
#             array(map(lambda i1, i2: cdist(X[[i1], :], X[[i2], :], metric=metric)[0][0],
#                       idx1, idx2))
#         weighted_distance = weights[idx1] * weights[idx2] * weighted_distance
#
#         permi = argsort(array(weighted_distance))  # get the pemutation that sorts by weighted_distance
#         # reduce until
#         # keep the reduce_to_npts weight_distance smallest pairs (reduce the
#         idx1 = idx1[permi][0]
#         idx2 = idx2[permi][0]
#
#     else:
#         return X, weights


def order_clus_idx_and_centers_by_decreasing_frequencies(clus_idx, clus_centers):
    """
    Returns a version of clus_idx and clus_centers where the clus_idx (indexing into corresponding clus_centers)
    are ordered according to their frequency of occurence.

    Note: If clus_idx has gaps (i.e. doesn't have all elements from 0 to n), the missing (i.e. not indexed by
    a clus_idx) clus_centers will be removed.
    """
    clus_idx_freqs = Counter(clus_idx)  # count the clus_idx occurences
    most_common_first_clus_idx = list(zip(*clus_idx_freqs.most_common()))[0]
    clus_idx_map = {k: v for k, v in zip(most_common_first_clus_idx, list(range(len(most_common_first_clus_idx))))}

    clus_idx = array([clus_idx_map[idx] for idx in clus_idx])
    clus_centers = clus_centers[array(most_common_first_clus_idx)]

    return clus_idx, clus_centers


def order_clus_idx_by_decreasing_frequencies(clus_idx):
    """
    Returns a version of clus_idx and clus_centers where the clus_idx (indexing into corresponding clus_centers)
    are ordered according to their frequency of occurence.

    Note: If clus_idx has gaps (i.e. doesn't have all elements from 0 to n), the missing (i.e. not indexed by
    a clus_idx) clus_centers will be removed.
    """
    clus_idx_freqs = Counter(clus_idx)  # count the clus_idx occurences
    most_common_first_clus_idx = list(zip(*clus_idx_freqs.most_common()))[0]
    clus_idx_map = {k: v for k, v in zip(most_common_first_clus_idx, list(range(len(most_common_first_clus_idx))))}

    clus_idx = array([clus_idx_map[idx] for idx in clus_idx])

    return clus_idx
