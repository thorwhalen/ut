"""Supervised clustering"""
__author__ = 'thor'

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster.k_means_ import _labels_inertia
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import row_norms

import numpy as np
import bisect

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from sklearn.base import ClusterMixin

from numpy import max, array, unique, ones, nan
from collections import defaultdict, Counter
import pandas as pd



def y_idx_dict(y):
    y_idx = defaultdict(list)
    list(map(lambda i, y_item: y_idx[y_item].append(i), *list(zip(*enumerate(y)))))
    return y_idx


def kmeans_per_y_dict(X, y, y_idx=None):
    if y_idx is None:
        y_idx = y_idx_dict(y)
    kmeans_for_y = dict()
    for yy, idx in y_idx.items():
        kmeans_for_y[yy] = KMeans(n_clusters=1).fit(X[idx, :])
    return kmeans_for_y


class SupervisedKMeans(ClusterMixin):
    """

    """
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters

    def fit(self, X, y):
        y_count = pd.Series(Counter(y)).sort_values(ascending=False)
        if len(y_count) > self.n_clusters:
            # if there's more unique ys than the requested labels, only take the most frequent ones for the fit
            hi_count_ys = y_count.iloc[:self.n_clusters].index.values
            is_a_hi_count_y_lidx = array(pd.Series(y).isin(hi_count_ys))
            XX = X[is_a_hi_count_y_lidx, :]
            yy = y[is_a_hi_count_y_lidx]
        else:
            XX = X
            yy = y

        y_idx = y_idx_dict(yy)
        n_ys = len(y_idx)
        kmeans_for_y = kmeans_per_y_dict(XX, yy, y_idx)
        inertias_for_y = {y: km.inertia_ for y, km in kmeans_for_y.items()}
        y_n_clusters_for_y = ones(n_ys) \
                             + _choose_distribution_according_to_weights(weights=list(inertias_for_y.values()),
                                                                         total_int_to_distribute=self.n_clusters - n_ys)
        y_n_clusters_for_y = y_n_clusters_for_y.astype(int)
        y_n_clusters_for_y = {y: x for y, x in zip(list(inertias_for_y.keys()), y_n_clusters_for_y)}

        centroids_ = list()
        km = dict()
        for this_y, idx in y_idx.items():
            km[this_y] = dict()
            km[this_y]['km'] = KMeans(n_clusters=y_n_clusters_for_y[this_y]).fit(XX[idx, :])
            km[this_y]['idx_offset'] = len(centroids_)
            centroids_.extend(list(km[this_y]['km'].cluster_centers_))

        self.cluster_centers_ = centroids_
        self.knn_ = NearestNeighbors(n_neighbors=1).fit(array(centroids_))
        self.km_for_y_ = km

        return self

    def predict(self, X, y=None):
        if y is None:
            return self.knn_.kneighbors(X, return_distance=False)[:, 0]
        else:
            y = array(y)
            cluster_idx = nan * ones(len(X))
            for yy in unique(y):
                lidx = y == yy
                if yy in self.km_for_y_:
                    cluster_idx[lidx] = self.km_for_y_[yy]['km'].predict(X[lidx, :]) + self.km_for_y_[yy]['idx_offset']
                else:
                    cluster_idx[lidx] = self.knn_.kneighbors(X[lidx, :], return_distance=False)[:, 0]
            return cluster_idx

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X, y)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_params(self):
        return self.__dict__


class SeperateClassKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self,
                 n_clusters=8,
                 method='volume',
                 clusterer=KMeans(),
                 keep_seperated_cluster_centroids=False):

        self.n_clusters = n_clusters
        self.method = method
        self.clusterer = clusterer
        self.keep_seperated_cluster_centroids = keep_seperated_cluster_centroids

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if self.n_clusters < n_classes:
            self.n_clusters = n_classes

        if isinstance(self.n_clusters, float):
            self.n_clusters = int(self.n_clusters)

        if isinstance(self.n_clusters, int):
            num_of_clusters_for_class = \
                _choose_class_weights(X, y, n_clusters=self.n_clusters, method=self.method, clusterer=self.clusterer)
        else:
            num_of_clusters_for_class = np.array(list(map(int, self.n_clusters)))
            self.n_clusters = int(np.sum(num_of_clusters_for_class))
            assert len(num_of_clusters_for_class) == n_classes, \
                "n_clusters should be a number or an array of #classes numbers"

        cluster_center_closeness = np.zeros(n_classes)
        cluster_centers_ = list()
        labels_ = list()
        if self.keep_seperated_cluster_centroids:
            self.class_centroids_ = dict()
            self.class_labels_ = dict()
        clusters_so_far = 0
        for i, label in enumerate(self.classes_):
            self.clusterer.set_params(n_clusters=int(num_of_clusters_for_class[i]))
            self.clusterer.fit(X[y == label])
            new_labels = clusters_so_far + self.clusterer.labels_
            labels_.extend(new_labels)
            clusters_so_far += num_of_clusters_for_class[i]
            cluster_centers_.extend(self.clusterer.cluster_centers_)
            if self.keep_seperated_cluster_centroids:
                self.class_centroids_[label] = self.clusterer.cluster_centers_
                self.class_labels_[label] = new_labels
            cluster_center_closeness[i] = self.clusterer.inertia_

        self.cluster_centers_ = np.array(cluster_centers_)
        self.labels_ = np.array(labels_)
        self.inertia_ = np.sum(cluster_center_closeness)

        return self

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        """
        return self.fit(X).labels_

    def fit_transform(self, X, y=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        return self.fit(X)._transform(X)

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'cluster_centers_')

        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        # X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return _labels_inertia(X, x_squared_norms, self.cluster_centers_)[0]


def _choose_class_weights(X, y, n_clusters=8, method='volume', clusterer=KMeans()):
    y_counts = Counter(y)
    classes_ = list(y_counts.keys())
    n_classes = len(classes_)
    n_clusters = int(n_clusters)
    if n_clusters < n_classes:
        n_clusters = n_classes
    if method == 'volume':
        ################################################################################################################
        # Take the number of pts per class to decide on how many clusters to get
        weights = list(y_counts.values())


    elif method == 'inertia':
        ################################################################################################################
        # Make a first pass with a single cluster for each class, and assign intertia to weights
        weights = np.zeros(n_classes)
        for i, label in enumerate(classes_):
            clusterer.set_params(n_clusters=1)
            clusterer.fit(X[y == label])
            weights[i] = clusterer.inertia_

    else:
        ValueError("Unknown method {}".format(method))

    weights = np.array(weights)
    # total_weight = np.sum(weights)
    # weight_covered_by_ones_blanket_ofor_each_class = total_weight * (n_clusters - n_classes) / float(n_clusters)
    weight_proportion_covered_by_blanket_of_ones = n_classes / float(n_clusters)
    remaining_weights = np.array([max(x, 0.0) for x in weights - weight_proportion_covered_by_blanket_of_ones])
    num_of_clusters_for_class = np.ones(n_classes) \
                                + _choose_distribution_according_to_weights(remaining_weights,
                                                                            n_clusters - n_classes)

    return num_of_clusters_for_class


def _choose_distribution_according_to_weights(weights, total_int_to_distribute):
    weights = np.array(list(map(float, weights))) / np.sum(weights)
    t = float(total_int_to_distribute) * weights
    distribution = np.floor(t)
    remaining_distribution = t - distribution
    remaining_int_to_distribute = total_int_to_distribute - np.sum(distribution)
    weighted_rand_generator = WeightedRandomGenerator(weights=remaining_distribution)
    for i in range(int(remaining_int_to_distribute)):
        idx = next(weighted_rand_generator)
        distribution[idx] += 1.0
    return distribution


class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = np.cumsum(weights)

    def __next__(self):
        rnd = np.random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return next(self)
