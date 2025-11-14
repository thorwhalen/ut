__author__ = 'thor'

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from pandas import DataFrame
import numpy as np
from nltk import word_tokenize
from functools import reduce


class HourOfDayTransformer(TransformerMixin):
    def __init__(self, date_field='datetime'):
        self.date_field = date_field

    def transform(self, X, **transform_params):
        hours = DataFrame(X[self.date_field].apply(lambda x: x.hour))
        return hours

    def fit(self, X, y=None, **fit_params):
        return self


class ModelTransformer(TransformerMixin):
    """
    Sometimes transformers do need to be fitted.
    ModelTransformer is used to wrap a scikit-learn model and make it behave like a transformer.
    This is useful when you want to use something like a KMeans clustering model to generate features for another model.
     It needs to be fitted in order to train the model it wraps.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))


class KVExtractor(TransformerMixin):
    """
    Transform multiple key/value columns in a scikit-learn pipeline.

    >>> import pandas as pd
    >>> D = pd.DataFrame([ ['a', 1, 'b', 2], ['b', 2, 'c', 3]], columns = ['k1', 'v1', 'k2', 'v2'])
    >>> kvpairs = [ ['k1', 'v1'], ['k2', 'v2'] ]
    >>> KVExtractor( kvpairs ).transform(D)
    [{'a': 1, 'b': 2}, {'c': 3, 'b': 2}]

    """

    def __init__(self, kvpairs):
        self.kpairs = kvpairs

    def transform(self, X, *_):
        result = []
        for index, rowdata in X.iterrows():
            rowdict = {}
            for kvp in self.kpairs:
                rowdict.update({rowdata[kvp[0]]: rowdata[kvp[1]]})
            result.append(rowdict)
        return result

    def fit(self, *_):
        return self


class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.keys]


class CategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        D = []
        for record in X.values:
            D.append({k: 1 for k in record[0]})
        return D


class AttributeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _flatten(self, d, parent_key='', sep='_'):
        """ Flatten dictonary
        """
        import collections

        items = []
        for k, v in list(d.items()):
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(list(self._flatten(v, new_key, sep=sep).items()))
            else:
                new_v = 1 if v == True else 0
                items.append((new_key, new_v))
        return dict(items)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        D = []
        for record in X.values:
            D.append(self._flatten(record[0]))
        return D


class KNNImputer(TransformerMixin):
    """
    Fill missing values using KNN Regressor
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        :param X: multidimensional numpy array like.
        """
        rows, features = X.shape

        mask = list([reduce(lambda h, t: h or t, x) for x in np.isnan(X)])
        criteria_for_bad = np.where(mask)[0]
        criteria_for_good = np.where(mask == np.zeros(len(mask)))[0]

        X_bad = X[criteria_for_bad]
        X_good = X[criteria_for_good]

        knn = KNeighborsRegressor(n_neighbors=self.k)

        for idx, x_bad in zip(criteria_for_bad.tolist(), X_bad):
            missing = np.isnan(x_bad)
            bad_dim = np.where(missing)[0]
            good_dim = np.where(missing == False)[0]

            for d in bad_dim:
                x = X_good[:, good_dim]
                y = X_good[:, d]
                knn.fit(x, y)

                X[idx, d] = knn.predict(x_bad[good_dim])

        return X


class NLTKBOW(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [{word: True for word in word_tokenize(document)} for document in X]
