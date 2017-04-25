from __future__ import division

import matplotlib.pyplot as plt


def comparison_vlines(y1, y2, c1='b', c2='r'):
    n = len(y1)
    assert len(y2) == n, "y1 and y2 need to be of the same length"
    plt.vlines(range(n), y1, y2)
    plt.plot(y1, 'o', color=c1)
    plt.plot(y2, 'o', color=c2)


def diff_comparison_vlines(y1, y2, c1='b', c2='k'):
    y = y1 - y2
    plt.vlines(range(len(y)), 0, y)
    plt.hlines(0, 0, len(y), colors=c2)
    plt.plot(range(len(y)), y, 'o', color=c1)


def ratio_comparison_vlines(y1, y2, c1='b', c2='k'):
    y = y1 / y2
    plt.vlines(range(len(y)), 1, y)
    plt.hlines(1, 0, len(y), colors=c2)
    plt.plot(range(len(y)), y, 'o', color=c1)
