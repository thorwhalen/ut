

from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import ClusterMixin

from numpy import max, array
from scipy.spatial.distance import cdist
import itertools


class DataBasedLabeling(ClusterMixin):
    pass


class KnnClusterDataBasedLabeling(DataBasedLabeling):
    def __init__(self,
                 knn_classifier=KNeighborsClassifier(n_neighbors=10),
                 label_proba_matrix_exponent=1,
                 clusterer=SpectralClustering(n_clusters=7)):
        self.knn_classifier = knn_classifier
        self.label_proba_matrix_exponent = label_proba_matrix_exponent
        if isinstance(clusterer, int):
            n_clusters = clusterer
            clusterer = SpectralClustering(n_clusters=n_clusters)
        self.clusterer = clusterer

    def fit(self, X, y):
        self.y_ = y
        self.knn_classifier.fit(X, self.y_)
        self.clusterer.fit(self.label_weights_matrix(X))

        return self

    def label_weights_matrix(self, X):
        label_weights_matrix = self.knn_classifier.predict_proba(X)
        if self.label_proba_matrix_exponent == 0:
            label_weights_matrix = (label_weights_matrix > 1 / len(self.knn_classifier.classes_)).astype(float)
        else:
            label_weights_matrix = label_weights_matrix ** self.label_proba_matrix_exponent

        return label_weights_matrix

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.clusterer.labels_


class BiDistanceDataBasedLabeling(DataBasedLabeling):
    def __init__(self,
                 n_labels=7,
                 label_distance_weight=0.5,
                 label_distance='equal',
                 feature_distance='euclidean',
                 agglomerative_clustering_kwargs={}.copy(),
                 save_merged_distance_mat=False):
        self.n_labels = n_labels
        self.label_distance_weight = label_distance_weight
        self.label_distance = label_distance
        self.feature_distance = feature_distance
        self.agglomerative_clustering_kwargs = agglomerative_clustering_kwargs
        self.save_merged_distance_mat = save_merged_distance_mat

    def fit(self, X, y):
        if self.label_distance_weight > 1:  # then normalize considering that feature weight is 1
            self.label_distance_weight = self.label_distance_weight / (self.label_distance_weight + 1)
        if isinstance(self.label_distance, str):
            if self.label_distance == 'equal':
                label_distance = lambda two_labels_tuple: float(two_labels_tuple[0] != two_labels_tuple[1])
            else:
                raise ValueError("Unknow label_distance: {}".format(self.label_distance))
        if isinstance(self.feature_distance, str):
            feature_distance = lambda pt_mat_1, pt_mat_2: cdist(pt_mat_1, pt_mat_2, metric=self.feature_distance)

        feature_dist_mat = feature_distance(X, X)
        feature_dist_mat /= max(feature_dist_mat)
        label_distance_mat = array(list(map(label_distance, itertools.product(y, y)))) \
            .reshape((len(y), len(y)))
        label_distance_mat /= max(label_distance_mat)

        merged_distance_mat = \
            self.label_distance_weight * label_distance_mat \
            + (1 - self.label_distance_weight) * feature_dist_mat

        self.clusterer_ = AgglomerativeClustering(n_clusters=self.n_labels,
                                                  affinity='precomputed',
                                                  linkage='complete',
                                                  **self.agglomerative_clustering_kwargs)
        self.clusterer_.fit(merged_distance_mat)
        if self.save_merged_distance_mat:
            self.merged_distance_mat_ = merged_distance_mat

        return self

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.clusterer_.labels_
