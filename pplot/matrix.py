__author__ = 'thor'

import matplotlib.patches as patches
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

from sklearn.cluster import MeanShift
import scipy.cluster.hierarchy as sch

import seaborn as sns


def plot_simil_mat_with_labels(simil_mat, y, inner_class_ordering='mean_shift_clusters', brightness=1.0, figsize=(10, 10)):
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
        idx = np.where(y_vals==y_val)[0]
        pt = idx[0] - 0.5
        s = idx[-1] - idx[0] + 1
        mids.append(pt + s / 2)
        ax.add_patch(patches.Rectangle(xy=(pt, pt), width=s, height=s, fill=False, linewidth=2, color='blue', alpha=0.5));
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
