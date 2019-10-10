

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils.validation import check_is_fitted
from scipy import linalg
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.utils.extmath import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model.base import LinearClassifierMixin

EPSILON = 1E-6


def class_freqs_df(y):
    cc_df = pd.DataFrame.from_dict(y).fillna(value=0.)
    # tot = cc_df.values.sum()
    res = cc_df.sum(axis=0)
    res /= res.sum()
    # res.columns = ['class_freqs']  # apparently, not needed

    return res


def rescale_dict(dictionary, scalar):
    res_dict = {}
    res_dict.update((key, value * float(scalar)) for (key, value) in list(dictionary.items()))
    return res_dict


def class_freqs(y, sample_weight):
    """Returns a dict of class frequencies
    """
    weighted_dict = []
    for scalar, dictionary in zip(sample_weight, y):
        weighted_dict.append(rescale_dict(dictionary, scalar))

    cc = Counter()
    for item in weighted_dict:
        cc.update(item)
    tot_weight = np.sum(sample_weight)
    return rescale_dict(cc, 1. / tot_weight)


def get_projection_dimension(X, tol=1e-08):
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    cov = pca.get_covariance()
    eigs, _ = np.linalg.eig(cov)
    n_dim_proj = np.sum(np.abs(eigs) > tol)

    return n_dim_proj


def _preprocess(X, y, sample_weight):
    all_labels = list(set().union(*(list(d.keys()) for d in y)))
    return pd.DataFrame(X), pd.DataFrame.from_dict(list(y)).fillna(value=0.), \
           pd.Series(sample_weight, name='X_weight'), \
           all_labels, \
           len(all_labels)


def lda_decision_function(class_freqs, class_means, covar, X):
    covar /= (X.shape[0] - class_means.shape[0])
    right_term = np.dot(np.linalg.inv(covar), class_means.T)
    linear_term = np.dot(X, right_term)
    bilin_term = np.diagonal(0.5 * np.dot(class_means, right_term))

    log_term = np.log(class_freqs)

    return linear_term - bilin_term + log_term


class FuzzyLDA(BaseEstimator, LinearClassifierMixin, TransformerMixin):
    def __init__(self, solver='eigen', n_components=None):
        self.solver = solver
        self.n_components = n_components

    def _solve_eigen(self, X, y, sample_weight):
        """
        Eigenvalue solver.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        sample_weight : array-like, shape (n_samples,)

        y : list of dicts, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,)
        """

        self.means_ = self._class_means(X, y, sample_weight)
        self.covariance_ = self.within_scatter_matrix(X, y, sample_weight)

        Sw = self.covariance_  # within scatter
        Sb = self.between_classes_scatter(X, y, sample_weight)  # between scatter

        if np.linalg.matrix_rank(Sw) < Sw.shape[0]:
            Sw += EPSILON * np.eye(Sw.shape[0])

        evals, evecs = linalg.eig(Sb, Sw, right=True)  # eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals)
                                                 )[::-1][:self._max_components]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
        evecs /= np.linalg.norm(evecs, axis=0)

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        # self.intercept_ = (-0.5 * np.diag(np.dot(self.means_, self.coef_.T)) +  TODO
        #                   np.log(self.priors_))

    def _class_means(self, X, y, sample_weight):
        """
        Compute weighted class means.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : list of dicts, shape (n_samples,)
            Labels.

        sample_weight : array-like (n_samples)
            Weights of the data points.
        Returns
        -------
        means : dict, of the form {'label': n_features}
            Weighted means for each class.
        """

        means = []
        for index, label in enumerate(self.all_labels):
            means.append(
                (self.X_df.mul(self.df_weights, axis=0)).mul(self.y_df[label], axis=0)[self.y_df[label] > 0.].mean(
                    axis=0))

        means_array = np.array(means)
        self.means_df = pd.DataFrame(means_array, index=self.all_labels)

        return means

    def _class_weights(self, X, y, sample_weight):
        """
        Compute total weights for each class.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : list of dicts, shape (n_samples,)
            Labels.

        sample_weight : array-like (n_samples)
            Weights of the data points.
        Returns
        -------
        class_weights : list
            Total weight for each class.
        """

        weights = []
        for index, label in enumerate(self.all_labels):
            weights.append(self.df_weights.mul(self.y_df[label], axis=0)[self.y_df[label] > 0.].sum(axis=0))

        weights_array = np.array(weights)

        return weights_array

    def within_scatter_matrix(self, X, y, sample_weight):
        """computes the within scatter matrix S_w
        """

        within_scatter_matrix = []
        for label in self.all_labels:
            within_scatter_matrix.append(
                np.cov((self.X_df[self.y_df[label] > 0.].mul(np.sqrt(self.df_weights[self.y_df[label] > 0.]), axis=0)) \
                       .mul(np.sqrt(self.y_df[label][self.y_df[label] > 0.]), axis=0).values.T, bias=1))

        return np.array(within_scatter_matrix).mean(axis=0)

    def within_scatter_matrix_list(self, X, y, sample_weight):
        """computes the within scatter matrix S_w
        """

        within_scatter_matrix = []
        for label in self.all_labels:
            within_scatter_matrix.append(
                np.cov((self.X_df[self.y_df[label] > 0.].mul(np.sqrt(self.df_weights[self.y_df[label] > 0.]), axis=0)) \
                       .mul(np.sqrt(self.y_df[label][self.y_df[label] > 0.]), axis=0).values.T, bias=1))

        return np.array(within_scatter_matrix)

    def between_classes_scatter(self, X, y, sample_weight):
        overall_mean = X.mean(axis=0)
        mean_vectors = pd.DataFrame(self._class_means(X, y, sample_weight), index=self.all_labels)
        mean_vectors -= overall_mean
        sq_weights = np.sqrt(self.class_weights_df)[0]

        res = mean_vectors.mul(np.sqrt(self.class_weights_df)[0], axis='index')

        Sb_list = []
        for label in self.all_labels:
            Sb_list.append(np.outer(res.loc[label].values, res.loc[label].values))
        # Sb = np.cov((res).values.T, bias =1)
        self.Sb = np.sum(Sb_list, axis=0)
        return np.sum(Sb_list, axis=0)

    def fit(self, X, y, sample_weight=None):
        """Fit LinearDiscriminantAnalysis model according to the given
           training data and parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : list of dicts, shape (n_samples,)
            Labels.

        sample_weight : array-like (n_samples)
            Weights of the data points.
        """

        if sample_weight is None:
            sample_weight = np.ones(len(y))

        if not isinstance(y[0], dict):  # check if the first element is a dict
            # if it's not, then go through every element and replace with {element: 1.0} if element not a dict
            new_y = list()
            for yy in y:
                if not isinstance(yy, dict):
                    new_y.append({yy: 1.0})
                else:
                    new_y.append(yy)
            y = np.array(new_y)

        self.classes_ = list(set().union(*(list(d.keys()) for d in y)))
        self.X_df, self.y_df, self.df_weights, self.all_labels, self.num_labels = \
            _preprocess(X, y, sample_weight)

        self.class_weights_df = pd.DataFrame(self._class_weights(X, y, sample_weight), index=self.all_labels)
        self.class_freqs_df = class_freqs_df(y)

        # Get the maximum number of components
        n_dimensions = get_projection_dimension(X)

        if self.n_components is None:
            self._max_components = min(len(self.classes_) - 1, n_dimensions)
        elif self.n_components <= len(self.classes_) - 1:
            self._max_components = min(self.n_components, n_dimensions)

        else:
            self._max_components = min(self.n_components, n_dimensions)
            self.extract_more_dim(X, y, sample_weight, self._max_components)

        if (self.solver == 'None' or self.solver == 'eigen'):
            self._solve_eigen(X, y, sample_weight)
        else:
            raise ValueError("unknown solver {} (valid solvers are None, "
                             "and 'eigen').".format(self.solver))
        if len(self.classes_) == 2:  # treat binary case as a special case
            self.coef_ = np.array(self.coef_[1, :] - self.coef_[0, :], ndmin=2)
            # self.intercept_ = np.array(self.intercept_[1] - self.intercept_[0],
            #                          ndmin=1)
        return self

    def transform(self, X):
        """Project data to maximize class separation.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, ['scalings_'], all_or_any=any)
        X_new = np.dot(X, self.scalings_)

        return X_new[:, :self._max_components]  # done

    def predict_proba(self, X):
        """Assign new point to a class.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        probas : array, shape (n_samples, n_classes)
            Predicted probas.
        """
        check_is_fitted(self, ['scalings_'], all_or_any=any)
        covar = np.array(self.covariance_)
        class_means = np.array(self.means_df)
        class_freqs = np.array(self.class_freqs_df)

        prob = lda_decision_function(class_freqs, class_means, covar, X)
        self.prob = prob  # Testing purposes
        # y_predict = np.argmax(lda_decision_function(class_freqs, class_means, covar, X), axis = 1)

        # np.exp(prob, prob)
        # prob += 1
        # np.reciprocal(prob, prob)
        if len(self.classes_) == 2:  # binary case
            np.exp(prob, prob)
            prob += 1
            np.reciprocal(prob, prob)
            return np.column_stack([1 - prob, prob])
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob = softmax(prob)
            return prob

    def predict(self, X, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        return np.argmax(self.predict_proba(X, sample_weight), axis=1)

    def extract_more_dim(self, X, y, sample_weight, n_dim):

        assert len(X[0]) >= n_dim, "n_dim cannot be larger than the number of features"
        n_labels = len(self.classes_)
        n_projections, remainder = divmod(n_dim, n_labels - 1)
        scalings = list()

        while n_projections > 0:
            n_projections -= 1
            FLDA = FuzzyLDA(n_components=n_labels - 1)
            FLDA.fit(X, y, sample_weight)
            X = X - np.dot(np.dot(X, FLDA.scalings_), np.transpose(FLDA.scalings_))
            scalings.append(FLDA.scalings_[:, :n_labels - 1])

        if remainder > 0:
            FLDA_remainder = FuzzyLDA(n_components=remainder)
            FLDA_remainder.fit(X, y, sample_weight)
            scalings.append(FLDA_remainder.scalings_[:, :remainder])

        self.scalings_ = np.hstack(scalings)


def sphericize(X, y, sample_weight=None):
    """Make the dataset spherical.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
            Input data.
    y : list of dicts, shape (n_samples,)
            Labels.
    sample_weight : array-like (n_samples)
            Weights of the data points.

    """

    if sample_weight is None:
        sample_weight = np.ones(len(y))

    fuz = FuzzyLDA().fit(X, y, sample_weight)
    W = fuz.within_scatter_matrix(X, y, sample_weight)
    eigenvals, eigenvecs = np.linalg.eig(W)
    D = np.diag(1 / np.sqrt(eigenvals))
    P = np.dot(eigenvecs, D)

    return np.dot(X, P)
