__author__ = 'thor'


import numpy as np
from numpy import *

from sklearn.qda import QDA as sk_QDA


class QDA(sk_QDA):
    def __init__(self, priors=None, reg_param=0.0):
        super(QDA, self).__init__(priors=priors, reg_param=reg_param)

    def transform(self, X):
        return self.predict_log_proba(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
