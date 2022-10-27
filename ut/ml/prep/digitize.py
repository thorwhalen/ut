__author__ = 'thor'

from numpy import *
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

from livestats.livestats import LiveStats


class IncrementalStats(BaseEstimator):
    def __init__(self, quantiles=None):
        self.quantiles = quantiles
        if self.quantiles is None:
            self.quantiles = 100
        if isinstance(self.quantiles, int):
            self.quantiles = linspace(start=0, stop=1, num=self.quantiles + 1)
            self.quantiles = self.quantiles[1:-1]

    def fit_partial(self, X):
        if not hasattr(self, 'column_stats_'):  # initialize livestats
            self.column_stats_ = [
                LiveStats(p=self.quantiles) for x in range(X.shape[1])
            ]
        if not hasattr(self, 'n_columns_'):
            self.n_columns_ = X.shape[1]
        for i, col in enumerate(X.T):
            list(filter(self.column_stats_[i].add, col))
        return self

    def quantiles_matrix(self):
        def get_quantiles_series(livestats_obj):
            quantiles = livestats_obj.quantiles()
            sr = pd.Series(
                data=[x[1] for x in quantiles], index=[x[0] for x in quantiles]
            )
            sr.sort_index(inplace=True)
            return sr

        qmat = pd.concat(list(map(get_quantiles_series, self.column_stats_)), axis=1)
        assert qmat.shape[0] == len(
            self.quantiles
        ), 'The concatination of the quantiles produced a different number of quantiles than it should'
        return qmat.as_matrix()


class DigitizeStream(BaseEstimator, TransformerMixin):
    def __init__(self, bins=None):
        self.bins = bins
        if self.bins is None:
            self.bins = 100
        if isinstance(self.bins, int):
            self.bins = linspace(start=0, stop=1, num=self.bins + 1)
            self.bins = self.bins[1:-1]

    def fit_partial(self, X, y=None):
        if not hasattr(self, 'column_learners_'):
            c = [Counter() for x in range(X.shape[1])]

        return self
