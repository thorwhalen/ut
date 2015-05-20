__author__ = 'thor'
"""
A collection of functions to generate synthetic data from real labeled data itself.
The purpose being to have data that resembles the real data, but with more or less force structure in order to
allow for controlled model assessment.
"""

from numpy import *


def label_means(X, y):
    XX = X.copy()
    for yy in unique(y):
        lidx = y == yy
        mean_X = mean(X[lidx, :], axis=0)
        XX[lidx, :] = mean_X
    return XX