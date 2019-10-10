__author__ = 'thor'

import matplotlib.patches as patches
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

from sklearn.cluster import MeanShift
import scipy.cluster.hierarchy as sch

import seaborn as sns


def xy_boxplot(X, y=None, col_labels=None, grid_size=None):
    if y is None:
        n_cols = X.shape[1]
        plt.boxplot(X)
        if col_labels is not None:
            assert n_cols == len(col_labels), "the number of items in col_labels should be equal to the num of cols"
            plt.xticks(list(range(1, n_cols + 1)), col_labels)
    else:
        unik_ys = np.unique(y)
        n_unik_ys = len(unik_ys)
        if grid_size is None:
            grid_size = (int(np.ceil(np.sqrt(n_unik_ys))), int(np.floor(np.sqrt(n_unik_ys))))
        for i, yy in enumerate(unik_ys, 1):
            lidx = y == yy
            XX = X[lidx, :]
            plt.subplot(grid_size[0], grid_size[1], i)
            xy_boxplot(XX, col_labels=col_labels)
            plt.gca().set_title(yy)


def vlines_ranges(X, aggr=('min', 'median', 'max'), axis=0, **kwargs):
    if isinstance(aggr, int):
        if aggr == 2:
            aggr = ('min', 'max')
        elif aggr == 3:
            aggr = ('min', 'median', 'max')
    assert len(aggr) >= 2, "aggr must have at least 2 elements"

    # aggr_func = list()
    # for i, a in enumerate(aggr):
    #     if isinstance(a, basestring):
    #         aggr_func.append(lambda x: getattr(np, a)(x, axis=axis))
    #     else:
    #         aggr_func.append(a)
    lo_val = getattr(np, aggr[0])(X, axis=axis)
    hi_val = getattr(np, aggr[-1])(X, axis=axis)
    x = np.arange(len(lo_val))
    plt.vlines(x, ymin=lo_val, ymax=hi_val, **kwargs)
    if len(aggr) > 2:
        markers = 'oxsd'
        for i, a in enumerate(aggr[1:-1]):
            plt.plot(x, getattr(np, a)(X, axis=axis), markers[i], **kwargs)


def vlines_of_matrix(X, y=None, col_labels=None, padding=0.05,
                     y_lim=None, col_label_rotation=0, ax=None, figsize=None, alpha=1):
    if y is None:
        if figsize is not None:
            plt.figure(figsize=figsize)
        if ax is None:
            ax = plt.gca()
        n_items, n_cols = X.shape
        if col_labels is not None:
            assert len(
                col_labels) == n_cols, "number of col_labels didn't match the number of columns in the input matrix"

        for i in range(n_cols):
            ax.vlines(np.linspace(i + padding, i + 1 - padding, n_items), 0, np.ravel(X[:, i]), colors='k', alpha=alpha)
            ax.hlines(0, i + padding, i + 1 - padding, colors='b', alpha=1)

        if y_lim is not None:
            ax.y_lim = plt.ylim(y_lim)
        else:
            ax.y_lim = plt.ylim()

        if col_labels is not None:
            plt.xticks(np.arange(n_cols) + 0.5, col_labels, rotation=col_label_rotation);
        else:
            plt.xticks([])

        ax.set_facecolor('w')
        plt.grid('off', axis='x')
    else:
        item_labels = np.unique(y)
        n_item_labels = len(item_labels)

        f, ax_list = plt.subplots(n_item_labels, sharex=True, sharey=True)
        if figsize is not None:
            if isinstance(figsize, (float, int)):
                figsize = (figsize, figsize)
            f.set_figwidth(figsize[0])
            f.set_figheight(figsize[1])

        for i, item_label in enumerate(item_labels):
            ax = ax_list[i]
            lidx = y == item_label
            x = X[lidx, :]
            vlines_of_matrix(x, col_labels=col_labels, y_lim=(0, 1), ax=ax, alpha=alpha)
            plt.yticks([])
            ax.set_ylabel(item_label)


def heatmap(X, y=None, col_labels=None, figsize=None, cmap=None, return_gcf=False, ax=None,
            xlabel_top=True, ylabel_left=True, xlabel_bottom=True, ylabel_right=True, **kwargs):
    n_items, n_cols = X.shape
    if col_labels is not None:
        if col_labels is not False:
            assert len(col_labels) == n_cols, \
                "col_labels length should be the same as the number of columns in the matrix"
    elif isinstance(X, pd.DataFrame):
        col_labels = list(X.columns)

    if figsize is None:
        x_size, y_size = X.shape
        if x_size >= y_size:
            figsize = (6, min(18, 6 * x_size / y_size))
        else:
            figsize = (min(18, 6 * y_size / x_size), 6)

    if cmap is None:
        if X.min(axis=0).min(axis=0) < 0:
            cmap = 'RdBu_r'
        else:
            cmap = 'hot_r'

    kwargs['cmap'] = cmap
    kwargs = dict(kwargs, interpolation='nearest', aspect='auto')

    if figsize is not False:
        plt.figure(figsize=figsize)

    if ax is None:
        plt.imshow(X, **kwargs)
    else:
        ax.imshow(X, **kwargs)
    plt.grid(None)

    if y is not None:
        y = np.array(y)
        assert all(sorted(y) == y), "This will only work if your row_labels are sorted"

        unik_ys, unik_ys_idx = np.unique(y, return_index=True)
        for u, i in zip(unik_ys, unik_ys_idx):
            plt.hlines(i - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
        plt.hlines(n_items - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
        plt.yticks(unik_ys_idx + np.diff(np.hstack((unik_ys_idx, n_items))) / 2, unik_ys)
    elif isinstance(X, pd.DataFrame):
        y_tick_labels = list(X.index)
        plt.yticks(list(range(len(y_tick_labels))), y_tick_labels);

    if col_labels is not None:
        plt.xticks(list(range(len(col_labels))), col_labels)
    else:
        plt.xticks([])

    plt.gca().xaxis.set_tick_params(labeltop=xlabel_top, labelbottom=xlabel_bottom)
    plt.gca().yaxis.set_tick_params(labelleft=ylabel_left, labelright=ylabel_right)

    if return_gcf:
        return plt.gcf()


def labeled_heatmap(X, y=None, col_labels=None):
    n_items, n_cols = X.shape
    if col_labels is not None:
        assert len(col_labels) == n_cols, "col_labels length should be the same as the number of columns in the matrix"

    heatmap(X, cmap='hot_r')
    plt.grid(None)

    assert all(sorted(y) == y), "This will only work if your row_labels are sorted"

    unik_ys, unik_ys_idx = np.unique(y, return_index=True)
    for u, i in zip(unik_ys, unik_ys_idx):
        plt.hlines(i - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
    plt.hlines(n_items - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
    plt.yticks(unik_ys_idx + np.diff(np.hstack((unik_ys_idx, n_items))) / 2, unik_ys)

    if col_labels is not None:
        assert len(col_labels) == n_cols, "col_labels length should be the same as the number of columns in the matrix"
        plt.xticks(list(range(len(col_labels))), col_labels);
    else:
        plt.xticks([])
    plt.gca().xaxis.set_tick_params(labeltop='on');


def plot_simil_mat_with_labels(simil_mat, y, inner_class_ordering='mean_shift_clusters', brightness=1.0,
                               figsize=(10, 10)):
    """
    A function that plots similarity matrices, grouping labels together and sorting by descending sum of similarities
    within a group.
    """
    simil_mat = simil_mat ** (1 / float(brightness))
    d = pd.DataFrame(simil_mat)
    d['y'] = y

    if inner_class_ordering == 'sum_simil':
        d['order'] = d.sum(axis=1)
    elif inner_class_ordering == 'mean_shift_clusters':
        d['order'] = np.nan
        for y_val in np.unique(y):
            lidx = y == y_val
            clus = MeanShift().fit(simil_mat[lidx][:, lidx])
            d['order'].iloc[lidx] = clus.labels_
    else:
        raise ValueError("Unknown inner_class_ordering")

    d = d.sort(['y', 'order'], ascending=False)
    y_vals = d['y']
    d = d.drop(labels=['y', 'order'], axis=1)

    permi = d.index.values
    w = simil_mat[permi][:, permi]

    plt.figure(figsize=figsize);
    ax = plt.gca();
    ax.matshow(w, cmap='gray_r');
    ax.grid(b=False)
    ax.set_aspect('equal', 'box');
    mids = list()
    unik_y_vals = np.unique(y_vals)
    for y_val in unik_y_vals:
        idx = np.where(y_vals == y_val)[0]
        pt = idx[0] - 0.5
        s = idx[-1] - idx[0] + 1
        mids.append(pt + s / 2)
        ax.add_patch(
            patches.Rectangle(xy=(pt, pt), width=s, height=s, fill=False, linewidth=2, color='blue', alpha=0.5));
    # plt.setp(ax.get_xticklabels(), visible=False);
    ax.xaxis.set_major_locator(plt.NullLocator())
    _ = ax.set_yticks(list(mids));
    _ = ax.set_yticklabels(unik_y_vals)
    _ = ax.set_xticks(list(mids));
    _ = ax.set_xticklabels(unik_y_vals, rotation=90)

    return y_vals.as_matrix()


def hierarchical_cluster_sorted_heatmap(df, only_return_sorted_df=False, seaborn_heatmap_kwargs=None):
    """
    A function to plot a square df (i.e. same indices and columns) that contains distances/similarities as it's values,
    as a heatmap whose indices and columns are sorted according to a hierarchical clustering
    (based on the distances listed in the df).
    :param df: The distance (or similarity) square matrix
    :param only_return_sorted_df: Default False. Set to True to return the df instead of the heatmap
    :param seaborn_heatmap_kwargs: the arguments to use in seaborn.heatmap (default is dict(cbar=False))
    :return: whatever sns.heatmap returns, or the sorted df if only_return_sorted_df=True
    """
    df = df.iloc[df.index.values, df.index.values]  # to make sure df is an index aligned square df
    Y = sch.linkage(np.array(df), method='centroid')
    Z = sch.dendrogram(Y, orientation='right', no_plot=True)
    index = np.array(Z['leaves'])
    df = df.iloc[index, index]
    if only_return_sorted_df:
        return df
    else:
        if seaborn_heatmap_kwargs is None:
            seaborn_heatmap_kwargs = {}
        seaborn_heatmap_kwargs = dict({"cbar": False}, **seaborn_heatmap_kwargs)
        return sns.heatmap(df, **seaborn_heatmap_kwargs)
