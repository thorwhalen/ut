from __future__ import division

from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.lda import LDA
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from numpy import array


__author__ = 'thor'



class FLDA(object):
    def __init__(self, pca_n_components=3, knn_n_neighs=10, n_scented_clusters=5, **kwargs):
        self.pca_n_components = pca_n_components
        self.knn_n_neighs = knn_n_neighs
        self.n_scented_clusters = n_scented_clusters
        self.lda_params = kwargs

    def fit(self, X, y):
        self.y_ = y
        self.scaler_ = StandardScaler()
        self.pca_ = PCA(n_components=self.pca_n_components)
        XX = self.pca_.fit_transform(self.scaler_.fit_transform(X))

        self.knn_ = KNeighborsClassifier(n_neighbors=self.knn_n_neighs)
        self.knn_.fit(XX, self.y_)

        yy = map(lambda nn: y[nn], self.knn_.kneighbors(XX)[1])
        self.cv_ = CountVectorizer(input='content', tokenizer=lambda x: x, lowercase=False)
        XXX = self.cv_.fit_transform(array(yy))
        self.tfidf_transformer_ = TfidfTransformer()
        XXX = self.tfidf_transformer_.fit_transform(XXX)

        self.clusterer_ = SpectralClustering(n_clusters=self.n_scented_clusters)
        yyy = self.clusterer_.fit_predict(XXX)

        self.lda_ = LDA(**self.lda_params)

        self.lda_.fit(XXX.todense(), yyy)

        return self

    def transform(self, X):
        #         return self.lda_.transform(self.pca_.fit_transform(self.scaler_.fit_transform(X)))
        X = self.pca_.transform(self.scaler_.transform(X))
        yy = map(lambda nn: self.y_[nn], self.knn_.kneighbors(X)[1])
        X = self.cv_.transform(array(yy))
        X = self.tfidf_transformer_.transform(X)
        return self.lda_.transform(X.todense())

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def set_params(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)


class FldaLite(FLDA):
    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        self.pca_ = PCA(n_components=self.pca_n_components)
        XX = self.pca_.fit_transform(self.scaler_.fit_transform(X))

        self.knn_ = KNeighborsClassifier(n_neighbors=self.knn_n_neighs)
        self.knn_.fit(XX, y)

        yy = map(lambda nn: y[nn], self.knn_.kneighbors(XX)[1])
        self.cv_ = CountVectorizer(input='content', tokenizer=lambda x: x, lowercase=False)
        XXX = self.cv_.fit_transform(array(yy))
        self.tfidf_transformer_ = TfidfTransformer()
        XXX = self.tfidf_transformer_.fit_transform(XXX)

        self.clusterer_ = SpectralClustering(n_clusters=self.n_scented_clusters)
        yyy = self.clusterer_.fit_predict(XXX)

        self.lda_ = LDA(**self.lda_params)
        self.lda_.fit(XX, yyy)

        return self

    def transform(self, X):
        return self.lda_.transform(self.pca_.fit_transform(self.scaler_.fit_transform(X)))
