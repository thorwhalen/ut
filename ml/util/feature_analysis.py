from __future__ import division

import matplotlib.pylab as plt
from numpy import array


def plot_feat_ranges(X, **kwargs):
    if isinstance(X[0], tuple) and len(X[0]) == 2:
        x_mins, x_maxs = zip(*X)
        n_feats = len(x_mins)
    else:
        X = array(X)
        x_mins = X.min(axis=0)
        x_maxs = X.max(axis=0)
        n_feats = X.shape[1]
    plt.figure(figsize=kwargs.pop('figsize', (12, 5)))
    plt.vlines(range(n_feats), x_mins, x_maxs, **kwargs)
    if n_feats < 40:
        plt.xticks(range(n_feats))
