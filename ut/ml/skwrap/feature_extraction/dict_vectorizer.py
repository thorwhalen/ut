__author__ = 'thor'

from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import six

import numpy as np
from pandas import DataFrame
from collections import Counter


class IterDictVectorizer(DictVectorizer):
    """Transforms lists of feature-value mappings or rows of a dataframe to vectors.

    It is like DictVectorizer (whose description was copied below), but:
    (1) works with pandas DataFrame X input (rows become feature-value mappings dict)
    (2) a minimum number of feature=value counts can be specified (by min_count)
    (3) The fit is faster than with DictVectorizer (at least with DataFrame input)

    This transformer turns lists of mappings (dict-like objects) of feature
    names to feature values into Numpy arrays or scipy.sparse matrices for use
    with scikit-learn estimators.

    When feature values are strings, this transformer will do a binary one-hot
    (aka one-of-K) coding: one boolean-valued feature is constructed for each
    of the possible string values that the feature can take on. For instance,
    a feature "f" that can take on the values "ham" and "spam" will become two
    features in the output, one signifying "f=ham", the other "f=spam".

    Features that do not occur in a sample (mapping) will have a zero value
    in the resulting array/matrix.

    Parameters
    ----------
    dtype : callable, optional
        The type of feature values. Passed to Numpy array/scipy.sparse matrix
        constructors as the dtype argument.
    separator: string, optional
        Separator string used when constructing new features for one-hot
        coding.
    sparse: boolean, optional.
        Whether transform should produce scipy.sparse matrices.
        True by default.
    sort: boolean, optional.
        Whether ``feature_names_`` and ``vocabulary_`` should be sorted when fitting.
        True by default.
    min_count: positive float or int:
        If min_count >= 1, min_count is the minimum number of feature=value count.
        If min_count < 1, min_count represent the minimum proportion of the data that should have feature=value

    Attributes
    ----------
    vocabulary_ : dict
        A dictionary mapping feature names to feature indices.

    feature_names_ : list
        A list of length n_features containing the feature names (e.g., "f=ham"
        and "f=spam").

    Examples
    --------
    >>> from sklearn.feature_extraction import DictVectorizer
    >>> v = DictVectorizer(sparse=False)
    >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
    >>> X = v.fit_transform(D)
    >>> X
    array([[ 2.,  0.,  1.],
           [ 0.,  1.,  3.]])
    >>> v.inverse_transform(X) == \
        [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
    True
    >>> v.transform({'foo': 4, 'unseen_feature': 3})
    array([[ 0.,  0.,  4.]])
    >>> from ut.ml.skwrap.feature_extraction import IterDictVectorizer
    >>> from pandas import DataFrame
    >>> v = IterDictVectorizer(sparse=False)
    >>> D = DataFrame([{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}])
    >>> X = v.fit_transform(D)
    >>> X
    array([[ 2.,  0.,  1.],
           [ 0.,  1.,  3.]])

    See also
    --------
    FeatureHasher : performs vectorization using only a hash function.
    sklearn.preprocessing.OneHotEncoder : handles nominal/categorical features
      encoded as columns of integers.
    """

    def __init__(
        self, dtype=np.float64, separator='=', sparse=True, sort=True, min_count=0
    ):
        self.dtype = dtype
        self.separator = separator
        self.sparse = sparse
        self.sort = sort
        self.min_count = min_count

    def fit(self, X, y=None):
        """Learn a list of feature name -> indices mappings.

        Parameters
        ----------
        X : Mapping or iterable over Mappings
            Dict(s) or Mapping(s) from feature names (arbitrary Python
            objects) to feature values (strings or convertible to dtype).
        y : (ignored)

        Returns
        -------
        self
        """
        feature_names = []
        vocab = {}

        feature_template = '{}' + self.separator + '{}'

        if isinstance(X, DataFrame):
            counts_of = dict()
            for col, val in X.items():
                counts_of[col] = Counter(val.dropna())
            self.feature_counts_ = {}
            _min_count = self.min_count
            if self.min_count < 1:
                _min_count *= len(X)
            else:
                _min_count = self.min_count
            self.df_columns_ = set()
            for k, v in counts_of.items():
                for kk, vv in v.items():
                    if vv >= _min_count:
                        self.feature_counts_[feature_template.format(k, kk)] = vv
                        self.df_columns_.add(k)
            feature_names = list(self.feature_counts_.keys())
        else:
            for x in X:
                for f, v in x.items():
                    if isinstance(v, str):
                        f = feature_template.format(f, v)
                    if f not in vocab:
                        feature_names.append(f)
                        vocab[f] = len(vocab)

        if self.sort:
            feature_names.sort()
            vocab = {f: i for i, f in enumerate(feature_names)}

        self.feature_names_ = feature_names
        self.vocabulary_ = vocab

        return self

    def transform(self, X, y=None):
        if isinstance(X, DataFrame):
            X = map(lambda x: x[1].dropna().to_dict(), X.iterrows())

        return super().transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class IterDictVectorizerWithText:
    def __init__(
        self,
        dtype=np.float64,
        separator='=',
        sparse=True,
        sort=True,
        min_count=0,
        text_vectorizers={},
    ):
        self.dict_vectorizer = IterDictVectorizer(
            dtype=dtype,
            separator=separator,
            sparse=sparse,
            sort=sort,
            min_count=min_count,
        )
        self.text_vectorizers = text_vectorizers

    def fit(self, X, y=None):
        # input validation
        assert isinstance(X, DataFrame), 'X must be a pandas DataFrame'
        if not set(self.text_vectorizers.keys()).issubset(X.columns):
            RuntimeError(
                'The following columns were specified in text_vectorizers, but were not in X:\n'
                + f'  {set(self.text_vectorizers.keys()).difference(X.columns)}'
            )

        # carry out the normal IterDictVectorizer.fit() for columns not in text_vectorizers
        self.dict_vectorizer_cols_ = set(X.columns).difference(
            list(self.text_vectorizers.keys())
        )
        self.dict_vectorizer.fit(X[self.dict_vectorizer_cols_])
        self.vocabulary_ = self.dict_vectorizer.vocabulary_

        # use the CounterVectorizers of text_vectorizers to fit the specified string columns
        for col in set(X.columns).intersection(list(self.text_vectorizers.keys())):
            self.text_vectorizers[col].fit(X[col])
            offset = len(self.vocabulary_)
            self.vocabulary_ = dict(
                self.vocabulary_,
                **{k: v + offset for k, v in self.text_vectorizers[col].items()}
            )

        self.feature_names_ = list(self.vocabulary_.keys())

    def transform(self, X, y=None):
        X1 = self.dict_vectorizer.transform(X[self.dict_vectorizer_cols_])
        X2 = np.hstack(
                map(
                    lambda col: self.text_vectorizers[col].transform(X[col]),
                    list(self.text_vectorizers.keys()),
                )
        )
        return np.hstack((X1, X2))
