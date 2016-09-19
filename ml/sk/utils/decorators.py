from __future__ import division

import types
from numpy import array, ndim

__author__ = 'thor'


def add_label_predicting_methods(self):
    """
    add methods to access predictions of labels

    Namely:
        self.predict_proba_with_labels(X) that returns a list of dicts of {label: predict_proba_score} entries
    and
        self.predict_proba_of_label(X, label) that returns an array of predict_proba number of a given label,
        or None if the label is not in self.classes_
    """

    def predict_proba_with_labels(self, X):
        if ndim(X) == 1:
            pred = self.predict_proba(X.reshape(1, -1))
            return dict(zip(self.classes_, array(pred)[0]))
        else:
            pred = self.predict_proba(X)
            return map(lambda row: dict(zip(self.classes_, row)), array(pred))

    def predict_proba_of_label(self, X, label):
        label_lidx = self.classes_ == label
        if any(label_lidx):
            if ndim(X) == 1:
                pred = self.predict_proba(X.reshape(1, -1))
                return array(pred)[0, label_lidx][0]
            else:
                pred = self.predict_proba(X)
                return array(pred[:, label_lidx])
        else:
            return None

    self.predict_proba_with_labels = types.MethodType(predict_proba_with_labels, self)
    self.predict_proba_of_label = types.MethodType(predict_proba_of_label, self)

    return self