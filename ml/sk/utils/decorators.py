from __future__ import division

from numpy import array, ndim, argsort
import sys

__author__ = 'thor'


non_methods_set = set(dir(sys.modules[__name__]))


def cumul_before_partial_fit(self, min_data_len):
    if not isinstance(min_data_len, (int, float)):
        if isinstance(min_data_len, basestring):
            if min_data_len in self.__dict__:
                min_data_len = self.__dict__[min_data_len]
            else:
                raise AttributeError("Your {} doesn't have attribute {}".format(
                    self.__class__, min_data_len
                ))
        elif callable(min_data_len):
            min_data_len = min_data_len(self)
        else:
            raise ValueError("Couldn't figure out the min_data_len")

    original_partial_fit = self.partial_fit
    cumul = list()
    got_enough_data = [False]

    def _cumul_before_partial_fit(X, *args, **kwargs):
        if got_enough_data[0]:
            original_partial_fit(X, *args, **kwargs)
        else:
            cumul.extend(map(list, X))
            if len(cumul) >= min_data_len:
                got_enough_data[0] = True
                original_partial_fit(array(cumul), *args, **kwargs)

    self.partial_fit = _cumul_before_partial_fit

    return self


def predict_proba_with_labels(self, X):
    """ returns a list of dicts of {label: predict_proba_score} entries """
    if ndim(X) == 1:
        pred = self.predict_proba(X.reshape(1, -1))
        return dict(zip(self.classes_, array(pred)[0]))
    else:
        pred = self.predict_proba(X)
        return map(lambda row: dict(zip(self.classes_, row)), array(pred))


def predict_proba_of_label(self, X, label):
    """
    If X is a single (ndim==1) feature vector, returns the probability of a given label according to the
    (predict_proba method of)
    If X is a observations x features matrix (ndarray), will return an array of probabilities for each observation
    If the label is not in self.classes_ will raise a LookupError
    """
    label_lidx = array(self.classes_) == label
    if any(label_lidx):
        if ndim(X) == 1:
            pred = self.predict_proba(X.reshape(1, -1))
            return array(pred)[0, label_lidx][0]
        else:
            pred = self.predict_proba(X)
            return array(pred[:, label_lidx]).reshape(-1)
    else:
        raise LookupError("The label {} wasn't found in the model")


def label_prob_argsort(self, X, label):
    """
    X is a observations x features matrix (ndarray) and label one of the labels the model modeled.
    The function will return an "argsort" array idx which will indicate how the input X can be sorted by decreasing
    probability of the given label.
    That is, such that
        self.predict_proba(X[label_prob_argsort(self, X, label), :])[:, self.classes_ == label].reshape(-1))
    will be monotone decreasing.
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.linear_model import LogisticRegressionCV
    >>> from numpy import diff, all
    >>> X, y = make_blobs()
    >>> clf = LogisticRegressionCV().fit(X, y)
    >>> label = clf.classes_[0]
    >>> permutation_idx = label_prob_argsort(clf, X, label)
    >>> sorted_predict_proba_matrix = clf.predict_proba(X)[permutation_idx, :]
    >>> assert all(diff(sorted_predict_proba_matrix[:, clf.classes_ == label].reshape(-1)) <= 0)
    """
    return argsort(predict_proba_of_label(self, X, label))[::-1]


def true_positive_rate(self, X, y):
    return sum(self.predict(X) == y) / float(len(y))

######### This is so that we can get a set of the  methods outside...

methods_set = set(dir(sys.modules[__name__])).difference(non_methods_set).difference({'non_methods_set'})


######### This is so that we can get an object that has these methods as attributes
######### (to make it easier to see what's available)

class Struct(object):
    def __init__(self, method_names):
        for method_name in method_names:
            setattr(self, method_name, getattr(sys.modules[__name__], method_name))


model_methods = Struct(methods_set)

