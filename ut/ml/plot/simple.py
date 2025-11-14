import numpy as np
import matplotlib.pylab as plt


def plot_cumul_explained_variance(model):
    """
    Takes a model object that has a explained_variance_ratio_ attribute (like PCA for e.g.) and plots the cumulative
    explained variance.
    :param model:
    :return:
    """
    n = len(model.explained_variance_ratio_)
    plt.plot(list(range(1, n + 1)), np.cumsum(model.explained_variance_ratio_), '-o')
    plt.xticks(list(range(1, n + 1)))
    plt.xlabel('Num of components')
    plt.ylabel('Cumulative explained Variance')


def add_correlation_line(line2d=None, xy_line=True, include_corr=True):
    if line2d is None:
        line2d = plt.gca().lines[0]
    elif isinstance(line2d, list):
        line2d = line2d.lines[0]

    x = line2d.get_xdata()
    y = line2d.get_ydata()

    m, b = np.polyfit(x, y, 1)

    if xy_line or include_corr:
        xy = list(x) + list(y)
        min_val = min(xy)
        max_val = max(xy)

        if xy_line:
            plt.plot([min_val, max_val], [min_val, max_val], 'k:')

        if include_corr:
            corr = np.corrcoef(x, y)[0, 1]
            plt.text(min_val * 1.01, max_val * 0.99, f'corr={corr:.4f}')

    plt.plot(x, m * x + b, 'k-')
