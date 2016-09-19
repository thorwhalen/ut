from __future__ import division

from metric_learn.lmnn import LMNN as _LMNN

__author__ = 'thor'


class LMNN(_LMNN):
    def fit_transform(self, X, y=None):
        return super(self.__class__, self).fit(X, y).transform(X)
