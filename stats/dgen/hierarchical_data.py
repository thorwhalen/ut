"""Generating hierarchical data"""

__author__ = 'thor'

from sklearn.datasets import make_blobs
from numpy import triu_indices, ndarray
from scipy.spatial.distance import cdist


def make_hblobs(n_samples=100,
                n_features=2,
                centers=(2, 3),
                cluster_std=1.0,
                center_box=(-10.0, 10.0),
                shuffle=True,
                random_state=None):
    """
    It uses the sklearn.datasets make_blobs to generate "hierarchical blobs".

    If you call it with an integer centers, it will just call make_blobs normally.

    But if you call it with a list of integer centers, it will call make_blobs recursively to choose centers
    centered on the centers of the last iteration.
    The actual number of centers that it will create is prod(centers).

    The cluster_std argument can also be a number or a list of numbers (of the same length as centers)
    If cluster_std is a number, it is immediately changed to be a list repeating that same number and then
    used as follows:
        * In the first step, we used cluster_std[0] as the cluster_std of the make_blobs() call
        * In every subsequent step, we multiply cluster_std[i] by the minimum distance between the centers

    Example:
        make_hblobs(n_samples=100, centers=[2, 3, 4])
    it will use make_blobs with with n_samples=2 to get 2 points.
    These points will then be passed, as centers of a next call to make_blobs to get 6 (=2*3, I swear!) points,
    3 centered on the first center, 3 centered on the seconds center.
    We now have 6 centers.
    We draw 4 points around each of these 6 centers to get 24 centers.
    Now that we have all our centers, we simply call the normal make_blobs with them (asking for n_samples=100))
    """

    if isinstance(centers, (list, tuple, ndarray)):
        if len(centers) == 1:
            return make_blobs(n_samples, n_features, centers[0], cluster_std, center_box, shuffle, random_state)
        else:
            # assert prod(centers) <= n_samples, "prod(centers) < n_samples !!!"
            if isinstance(cluster_std, (float, int)):
                cluster_std = [cluster_std] * len(centers)
            assert len(cluster_std) == len(centers)

            level_centers, _ = make_blobs(n_samples=centers[0], n_features=n_features, centers=centers[0],
                                          cluster_std=cluster_std[0],
                                          center_box=center_box, shuffle=shuffle, random_state=random_state)

            for this_center, _cluster_std in zip(centers[1:], cluster_std[1:]):
                n = this_center * len(level_centers)
                min_dist = cdist(level_centers, level_centers)[triu_indices(len(level_centers), k=1)].min()
                level_centers, y = make_blobs(n_samples=min(n_samples, n),
                                              n_features=n_features,
                                              centers=level_centers,
                                              cluster_std=min_dist * _cluster_std,
                                              center_box=center_box, shuffle=shuffle, random_state=random_state)

        min_dist = cdist(level_centers, level_centers)[triu_indices(len(level_centers), k=1)].min()

        return make_blobs(n_samples=n_samples, n_features=n_features, centers=level_centers,
                          cluster_std=min_dist * cluster_std[-1],
                          center_box=center_box, shuffle=shuffle, random_state=random_state)

    else:
        return make_blobs(n_samples, n_features, centers, cluster_std, center_box, shuffle, random_state)



