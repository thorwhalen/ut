__author__ = 'thor'

import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import finance
from matplotlib.collections import LineCollection
import pandas as pd

from sklearn import cluster, covariance, manifold


def affinity_propagation_network(X, names=None):
    """
    Cluster (affinity propagation based on the correlation of ) rows of X,
        printing out cluster contents and
        drawing a labeled network of the results, with darker edges for more correlated pairs

    X can be an array or a pandas DataFrame.
    names are labels for the rows, which will be taken to be the indices of the dataframe, or the "names" column, or
    0..n-1 otherwise

    Very lightly adapted from
        http://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#example-applications-plot-stock-market-py

    Author: Gael Varoquaux gael.varoquaux@normalesup.org
    License: BSD 3 clause

    The output of the 3 models are combined in a 2D graph where nodes represents the columns and edges the:
    * cluster labels are used to define the color of the nodes
    * the sparse covariance model is used to display the strength of the edges
    * the 2D embedding is used to position the nodes in the plan

    This example has a fair amount of visualization-related code, as visualization is crucial here to display the graph.
    One of the challenge is to position the labels minimizing overlap. For this we use an heuristic based on the
    direction of the nearest neighbor along each axis
    """

    X = X.copy()

    if isinstance(X, pd.DataFrame):
        if isinstance(names, str):
            names = X.pop(names)
        elif names is None:
            names = X.index.values
        X = X.as_matrix().T
    elif names is None:
        names = list(range(X.shape[0]))

    ###############################################################################
    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    # X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)

    ###############################################################################
    # Cluster using affinity propagation

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print(('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i]))))

    ###############################################################################
    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane

    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    ###############################################################################
    # Visualization
    plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    #a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())

    plt.show()

