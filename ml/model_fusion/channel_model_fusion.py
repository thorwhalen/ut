from __future__ import division

from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV


class ClassifierFusion(BaseEstimator, ClassifierMixin):
    def __init__(self, model_for_channel):
        self.model_for_channel = model_for_channel

    @classmethod
    def for_same_model_class(cls, model_class, channels):
        return cls(model_for_channel={channel: model_class() for channel in channels})

    def fit(self, x_for_channel, y):
        for channel in x_for_channel.keys():
            if channel in self.model_for_channel:
                self.model_for_channel[channel].fit(x_for_channel[channel], y)
        return self

    def predict_proba_for_channel(self, x_for_channel):
        pp = dict()
        for channel in x_for_channel.keys():
            if channel in self.model_for_channel:
                pp[channel] = self.model_for_channel[channel].predict_proba(x_for_channel[channel])
        return pp

    def predict_proba(self, x_for_channel):
        pp = list()
        for p in self.predict_proba_for_channel(x_for_channel).values():
            pp.append(p)
        # pp.append(p / p.sum(axis=1)[:, None])
        pp = np.exp(np.array(map(np.log, np.array(pp))).sum(axis=0))
        return pp / pp.sum(axis=1)[:, None]

    def predict(self, x_for_channel):
        predict_probas = self.predict_proba(x_for_channel)
        return np.ravel(self.classes_[np.array(np.argmax(predict_probas, axis=1))])


def most_common(x):
    _most_common = Counter(x).most_common()
    n = len(_most_common)
    if n > 0:
        if n == 1 or (_most_common[0][1] > _most_common[1][1]):
            return _most_common[0][0]
        else:
            return np.random.choice([x[0] for x in most_common], size=1, p=[x[1] for x in most_common])
    else:
        return _most_common[0][0]


class VoteFusion(ClassifierFusion):
    def predict(self, x_for_channel):
        preds_for_channel = {channel: self.model_for_channel[channel].predict(x_for_channel[channel])
                             for channel in x_for_channel}

        predictions = np.array(map(list, preds_for_channel.values()))
        return np.array(map(most_common, zip(*predictions)))


class LogisticFusion(ClassifierFusion):
    def _x_for_logistic(self, x_for_channel):
        x_for_logistic = list()
        for channel, X in x_for_channel.iteritems():
            pp = self.model_for_channel[channel].predict_proba(X)
            x_for_logistic.append(pp[:, :-1])
        return np.hstack(x_for_logistic)

    def fit_fusion_with_already_fitted_models(self, x_for_channel, y):
        x_for_logistic = self._x_for_logistic(x_for_channel)
        self.lr = LogisticRegressionCV(multi_class='ovr').fit(x_for_logistic, y)
        return self

    def fit(self, x_for_channel, y):
        super(self.__class__, self).fit(x_for_channel, y)
        return self.fit_fusion_with_already_fitted_models(x_for_channel, y)

    def predict_proba(self, x_for_channel):
        x_for_logistic = self._x_for_logistic(x_for_channel)
        pp = self.lr.predict_proba(x_for_logistic)
        return (pp.T / pp.sum(axis=1).T).T

    def predict(self, x_for_channel):
        x_for_logistic = self._x_for_logistic(x_for_channel)
        return self.lr.predict(x_for_logistic)
