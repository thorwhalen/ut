import matplotlib.pyplot as plt
import numpy as np
from numpy import array


def _sort_y1_and_y2_according_to_diff(y1, y2):
    zipped = zip(y1, y2)
    zipped.sort(key=lambda x: x[0] - x[1])
    return zip(*zipped)


def comparison_vlines(y1, y2, order_wrt_diff=False, c1='b', c2='r', color_line=True, **kwargs):
    if order_wrt_diff:
        y1, y2 = _sort_y1_and_y2_according_to_diff(y1, y2)
    y1 = array(y1)
    y2 = array(y2)
    n = len(y1)
    assert len(y2) == n, "y1 and y2 need to be of the same length"
    if not color_line:
        plt.vlines(list(range(n)), y1, y2)
    else:
        a = y1 - y2
        pos_lidx = a >= 0
        pos_idx = np.where(pos_lidx)[0]
        neg_idx = np.where(~pos_lidx)[0]
        x_idx = np.arange(len(a))
        print('asd')
        plt.vlines(x_idx[pos_idx], y1[pos_idx], y2[pos_idx], colors=c1, **kwargs)
        plt.vlines(x_idx[neg_idx], y1[neg_idx], y2[neg_idx], colors=c2, **kwargs)
    plt.plot(y1, 'o', color=c1)
    plt.plot(y2, 'o', color=c2)


def diff_comparison_vlines(y1, y2, order_wrt_diff=False, c1='b', c2='k'):
    """

    :param y1:
    :param y2:
    :param c1:
    :param c2:
    :return: what plt.plot returns
    """
    if order_wrt_diff:
        y1, y2 = _sort_y1_and_y2_according_to_diff(y1, y2)
    y = array(y1) - array(y2)
    plt.vlines(list(range(len(y))), 0, y)
    plt.hlines(0, 0, len(y) - 1, colors=c2)
    return plt.plot(list(range(len(y))), y, 'o', color=c1)


def ratio_comparison_vlines(y1, y2, order_wrt_diff=False, c1='b', c2='k'):
    """
    Plots vlines of y1/y2.
    :param y1: numerator
    :param y2: denominator
    :param c1: color of numerator
    :param c2: color of denominator (will be a straight horizontal line placed at 1)
    :return: what plt.plot returns
    """
    if order_wrt_diff:
        y1, y2 = _sort_y1_and_y2_according_to_diff(y1, y2)
    y = array(y1) / array(y2)
    plt.vlines(list(range(len(y))), 1, y)
    plt.hlines(1, 0, len(y) - 1, colors=c2)
    return plt.plot(list(range(len(y))), y, 'o', color=c1)


def thresholded_vlines(a, thresh=0, pos_color='b', neg_color='r',
                       linestyles='solid', label='', data=None, **kwargs):
    a = np.array(a)
    pos_lidx = a >= thresh
    pos_idx = np.where(pos_lidx)[0]
    neg_idx = np.where(~pos_lidx)[0]
    x_idx = np.arange(len(a))
    plt.vlines(x_idx[pos_idx], thresh, a[pos_idx],
               colors=pos_color, linestyles=linestyles, label=label, data=data, **kwargs)
    plt.vlines(x_idx[neg_idx], thresh, a[neg_idx],
               colors=neg_color, linestyles=linestyles, label=label, data=data, **kwargs)


def side_by_side_bar(list_of_values_for_bars, width=1, spacing=1, names=None, colors=None):
    """
    A plotting utility making side by side bar graphs from a list of list (of same length) of values.

    :param list_of_values_for_bars:
    :param width: the width of the bar graphs
    :param spacing: the size of the spacing between groups of bars
    :param names: the names to assign to the bars, in the same order as in list_of_values_for_bars
    :param list_colors: the colors to use for the bars, in the same order as in list_of_values_for_bars
    :return: a nice plot!
    """

    # if no list_names is specified, the names on the legend will be integer
    if names is None:
        names = range(len(list_of_values_for_bars))
    # if no list_colors is specified, the colors will be chosen from the rainbow
    n_bars = len(list_of_values_for_bars)
    if colors is None:
        colors = plt.cm.rainbow(np.linspace(0, 1, n_bars))
    else:
        assert len(colors) >= n_bars, "There's not enough colors for the number of bars ({})".format(n_bars)
    ax = plt.subplot(111)
    # making each of the bar plot
    for i, list_of_values_for_bars in enumerate(list_of_values_for_bars):
        x = [width * j * n_bars + spacing * j + i * width for j in range(len(list_of_values_for_bars))]
        ax.bar(x, list_of_values_for_bars, width=width, color=colors[i], align='center')
    ax.legend(names)
    ax.xaxis.set_ticklabels([])
