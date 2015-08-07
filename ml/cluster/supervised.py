__author__ = 'thor'

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import bisect


class SeperateClassKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1):
        if hasattr(init, '__array__'):
            n_clusters = init.shape[0]
            init = np.asarray(init, dtype=np.float64)

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y_counter = Counter(y)
        self.classes_ = y_counter.keys()
        n_pts_per_class = y_counter.values()
        n_classes = len(self.classes_)
        if self.n_clusters < n_classes:
            self.n_clusters = n_classes

        ################################################################################################################
        # Take the number of pts per class to decide on how many clusters to get
        num_of_clusters_for_class = np.ones(n_classes) \
                                    + _choose_distribution_according_to_weights(n_pts_per_class, self.n_clusters)
        cluster_center_closeness = np.zeros(n_classes)
        cluster_centers_ = list()
        for i, label in enumerate(self.classes_):
            kmeans = KMeans(
                n_clusters=int(num_of_clusters_for_class[i]),
                init=self.init,
                max_iter=self.max_iter,
                tol=self.tol,
                precompute_distances=self.precompute_distances,
                n_init=self.n_init,
                verbose=self.verbose,
                random_state=self.random_state,
                copy_x=self.copy_x,
                n_jobs=self.n_jobs
            ).fit(X[y == label])
            cluster_centers_.extend(kmeans.cluster_centers_)
            cluster_center_closeness[i] = kmeans.inertia_

        self.cluster_centers_ = np.array(cluster_centers_)
        self.inertia_ = np.sum(cluster_center_closeness)
        return self
        # ################################################################################################################
        # # Now use the cluster_center_closeness decide on how many clusters to get
        #
        # weighted_rand_generator = WeightedRandomGenerator(weights=cluster_center_closeness)
        # n_clusters_for_label = np.repeat(1, repeats=n_classes)
        # for idx in weighted_rand_generator.next():
        #     n_clusters_for_label[idx] += 1

    def fit_predict(self, X, y):
        self.fit(X, y)
        self.predict(X)
        return self


def _choose_distribution_according_to_weights(weights, total_int_to_distribute):
    weights = np.array(map(float, weights)) / np.sum(weights)
    t = float(total_int_to_distribute) * weights
    distribution = np.floor(t)
    remaining_distribution = t - distribution
    remaining_int_to_distribute = total_int_to_distribute - np.sum(distribution)
    weighted_rand_generator = WeightedRandomGenerator(weights=remaining_distribution)
    for i in xrange(int(remaining_int_to_distribute)):
        idx = weighted_rand_generator.next()
        distribution[idx] += 1.0
    return distribution


class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = np.cumsum(weights)

    def next(self):
        rnd = np.random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()

