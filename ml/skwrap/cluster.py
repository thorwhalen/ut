__author__ = 'thor'

import numpy as np
from numpy import *

from sklearn.cluster import SpectralClustering as sk_SpectralClustering
from sklearn.cluster import MiniBatchKMeans as MiniBatchKMeans_sk
from sklearn.neighbors import NearestNeighbors

from numpy import vstack
from sklearn.cluster import KMeans
from ut.ml.utils import get_model_attributes


class SpectralClustering(sk_SpectralClustering):
    def __init__(
        self,
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
        kernel_params=None,
    ):
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
            kernel_params=kernel_params,
        )

    def fit(self, X, y=None):
        super(SpectralClustering, self).fit(X=X, y=y)
        self.cluster_centers_ = vstack(
            [mean(X[self.labels_ == i], axis=0) for i in sorted(unique(self.labels_))]
        )


class MiniBatchKMeans(MiniBatchKMeans_sk):
    def __init__(
        self,
        n_clusters=8,
        init='k-means++',
        max_iter=100,
        batch_size=100,
        verbose=0,
        compute_labels=False,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01,
        X_cumul=None,
    ):
        super(MiniBatchKMeans, self).__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            compute_labels=compute_labels,
            random_state=random_state,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=init_size,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
        )
        self.verbose = verbose
        self.X_cumul = X_cumul

    def partial_fit(self, X, y=None):
        if not hasattr(self, 'n_partial_fit_calls_'):
            self.n_partial_fit_calls_ = 0
            self.n_partial_fit_calls_actually_fitted_ = 0
            self.n_data_points_fitted_ = 0

        self.n_partial_fit_calls_ += 1

        if self.X_cumul is not None:
            self.X_cumul = vstack((self.X_cumul, X))
        else:
            self.X_cumul = X

        n_samples, n_features = self.X_cumul.shape
        if n_samples > self.n_clusters:
            super(MiniBatchKMeans, self).partial_fit(X=self.X_cumul, y=y)
            self.n_partial_fit_calls_actually_fitted_ += 1
            self.n_data_points_fitted_ += len(X)
            self.X_cumul = None

    def sort_model_params_according_to_decreasing_counts(self):
        permi = argsort(self.counts_)[::-1][: len(self.counts_)]
        self.counts_ = self.counts_[permi]
        self.cluster_centers_ = self.cluster_centers_[permi, :]

    def get_kmeans_object(self):
        kmeans_obj = KMeans()
        kmeans_obj.cluster_centers_ = self.cluster_centers_
        if hasattr(kmeans_obj, 'count_'):
            kmeans_obj.count_ = kmeans_obj.count_
        kmeans_obj.inverse_transform = lambda cluster_idx: kmeans_obj.cluster_centers_[
            cluster_idx, :
        ]
        return kmeans_obj

    def __getstate__(self):
        state = get_model_attributes(
            self, model_name_as_dict_root=False, exclude=('random_state_',)
        )
        if hasattr(self, 'verbose'):
            state['verbose'] = self.verbose
        if hasattr(self, 'verbose'):
            state['batch_size'] = self.batch_size
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            self.__setattr__(k, v)
        if hasattr(self, 'verbose'):
            self.__setattr__('verbose', self.verbose)
        if hasattr(self, 'batch_size'):
            self.__setattr__('batch_size', self.batch_size)


class MiniBatchKMeansTransformer(MiniBatchKMeans):
    def __init__(
        self,
        n_clusters=8,
        init='k-means++',
        max_iter=100,
        batch_size=100,
        verbose=0,
        compute_labels=False,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01,
        X_cumul=None,
    ):
        super(self.__class__, self).__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            compute_labels=compute_labels,
            random_state=random_state,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=init_size,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
            X_cumul=None,
        )

    def transform(self, X):
        return super(self.__class__, self).predict(X)

    def centroid_distances(self, X, y=None):
        return super(self.__class__, self).transform(X, y)


class MiniBatchKMeansTransformerWithKnn(MiniBatchKMeans):
    def __init__(
        self,
        n_clusters=8,
        init='k-means++',
        max_iter=100,
        batch_size=100,
        verbose=0,
        compute_labels=False,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01,
        X_cumul=None,
    ):
        super(self.__class__, self).__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
            compute_labels=compute_labels,
            random_state=random_state,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=init_size,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio,
            X_cumul=None,
        )
        self.knn = None

    def transform(self, X):
        if self.knn is None:
            self.knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(
                self.cluster_centers_
            )
        return reshape(self.knn.kneighbors(X)[1], (len(X),))

    def centroid_distances(self, X, y=None):
        return super(self.__class__, self).transform(X, y)
