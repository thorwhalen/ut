from __future__ import division

from numpy import sum, hstack, cumsum
from sklearn.datasets import make_blobs


def make_multimodal_blobs(tag_centers=(2, 2), n_samples=100, n_features=2, cluster_std=1.0, center_box=(-10.0, 10.0),
                          shuffle=True, random_state=None):
    if isinstance(tag_centers, int):
        tag_centers = int(n_samples // tag_centers) * [30]
    centers = sum(tag_centers)
    if n_samples is None:
        n_samples = 30 * centers
    assert centers <= n_samples, "centers <= n_samples"
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                      cluster_std=cluster_std, center_box=center_box, shuffle=shuffle, random_state=random_state)

    tag_centers_intervals = hstack((0, cumsum(tag_centers)))
    lidx = list()
    for i in range(len(tag_centers)):
        lidx.append((y >= tag_centers_intervals[i]) & (y < tag_centers_intervals[i + 1]))
    for i in range(len(lidx)):
        y[lidx[i]] = i
    return X, y