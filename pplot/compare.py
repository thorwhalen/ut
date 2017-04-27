from __future__ import division

import matplotlib.pyplot as plt
from numpy import array


def comparison_vlines(y1, y2, c1='b', c2='r'):
    n = len(y1)
    assert len(y2) == n, "y1 and y2 need to be of the same length"
    plt.vlines(range(n), y1, y2)
    plt.plot(y1, 'o', color=c1)
    plt.plot(y2, 'o', color=c2)


def diff_comparison_vlines(y1, y2, c1='b', c2='k'):
    """

    :param y1:
    :param y2:
    :param c1:
    :param c2:
    :return: what plt.plot returns
    """
    y = array(y1) - array(y2)
    plt.vlines(range(len(y)), 0, y)
    plt.hlines(0, 0, len(y) - 1, colors=c2)
    return plt.plot(range(len(y)), y, 'o', color=c1)


def ratio_comparison_vlines(y1, y2, c1='b', c2='k'):
    """
    Plots vlines of y1/y2.
    :param y1: numerator
    :param y2: denominator
    :param c1: color of numerator
    :param c2: color of denominator (will be a straight horizontal line placed at 1)
    :return: what plt.plot returns
    """
    y = array(y1) / array(y2)
    plt.vlines(range(len(y)), 1, y)
    plt.hlines(1, 0, len(y) - 1, colors=c2)
    return plt.plot(range(len(y)), y, 'o', color=c1)
