__author__ = 'thor'

from numpy import zeros, argmin, array, unique, where
from scipy.spatial.distance import cdist


def _df_picker(df, x_cols, y_col):
    return df[x_cols].as_matrix(), df[[y_col]].as_matrix()


def df_picker_data_prep(x_cols, y_col):
    return lambda df: _df_picker(df, x_cols, y_col)


def binomial_probs_to_multinomial_probs(binomial_probs):
    multinomial_probs = zeros((len(binomial_probs), 2))
    multinomial_probs[:, 1] = binomial_probs
    multinomial_probs[:, 0] = 1 - multinomial_probs[:, 1]
    return multinomial_probs


def multinomial_probs_to_binomial_probs(multinomial_probs):
    return multinomial_probs[:, 1]


def normalize_to_one(arr):
    arr = array(arr)
    return arr / float(sum(arr))


def point_idx_closest_to_centroid(X, centroid=None):
    """
    X is a n x m ndarray of n points in m dimensions.
    point_closest_to_centroid(X, centroid) returns the index of the point of X (a row of X) that is closest to the given
    centroid point. If centroid is not given, the actual centroid, X.mean(axis=1), is taken.
    """
    if centroid is None:
        centroid = X.mean(axis=0)
    return argmin(cdist(X, [centroid])[:, 0])


def point_closest_to_centroid(X, centroid=None):
    """
    X is a n x m ndarray of n points in m dimensions.
    point_closest_to_centroid(X, centroid) returns the point of X (a row of X) that is closest to the given
    centroid point. If centroid is not given, the actual centroid, X.mean(axis=1), is taken.
    """
    if centroid is None:
        centroid = X.mean(axis=0)
    return X[argmin(cdist(X, [centroid])[:, 0])]


def get_cluster_representatives_indices(X, clus_labels, clus_centers):
    representative_indices = list()
    for label in unique(clus_labels):
        cluster_point_indices = where(clus_labels == label)[0]
        min_idx = argmin(
            cdist(X[cluster_point_indices, :], [clus_centers[label, :]])[:, 0]
        )
        representative_indices.append(cluster_point_indices[min_idx])
    return array(representative_indices)
