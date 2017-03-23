__author__ = 'thor'

import pandas as pd
from numpy import array, random
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from collections import Counter


def balanced_data(X, y, method='sample'):
    """
    Get get subset of X (and aligned y) such that all y elements have the same count.
    (i.e. all(Counter(y) == min(Counter(y).values())))

    :param X: data matrix
    :param y: label array (number aligned with rows of X)
    :param method:
        "sample" (will choose sample by random sampling without replacement)
        "knn" (will choose the nearest neighbors of the centroids of each label subset -- much more representative,
        but also a lot lot slower)

    :return: A subset of X and aligned y
    """
    yc = Counter(y)
    min_yc = min(yc.values())
    new_X = list()
    new_y = list()
    for yi in yc.keys():
        yi_lidx = y == yi
        XX = X[yi_lidx, :]
        yy = y[yi_lidx]
        if yc[yi] != min_yc:
            if method == 'knn':
                km = KMeans(n_clusters=min_yc).fit(XX)
                knn_idx, _ = vq(XX, km.cluster_centers_)
            else:
                knn_idx = random.choice(len(XX), min_yc, replace=False)
            new_X.extend(XX[knn_idx, :].tolist())
            new_y.extend(yy[knn_idx])
        else:
            new_X.extend(XX.tolist())
            new_y.extend(yy.tolist())

    return array(new_X), array(new_y)


def label_balanced_subset(data, label, random_seed=None):
    """
    An newer version of this can be found in balanced_data.

    Get get subset of X (and aligned y) such that all y elements have the same count.
    (i.e. all(Counter(y) == min(Counter(y).values())))

    :param data: data matrix or dataframe containing the data and the labels
    :param label: label array (number aligned with rows of data) or name of the column where labels are in a dataframe
        data input.
    :param random_seed: seed for the random selection
    :return: A subset of X and aligned y
    """
    if not isinstance(data, pd.DataFrame) or label not in data.columns:
        # ... then assume data is the X and label is the y arrays of the supervised learning setup
        return_arrays = True
        dg = pd.concat([pd.DataFrame(data), pd.DataFrame({'label': label})], axis=1).groupby('label')
    else:
        return_arrays = False
        # ... then assume data contains both explanatory and label (targets of classification) data
        dg = data.groupby(label)

    min_count = min(dg.size())

    def get_subset_data_idx(x, random_seed):
        if random_seed == -1:
            return slice(0, min_count)
        else:
            random.seed(random_seed)
            return random.choice(a=len(x), size=min_count, replace=False)

    subset_data = pd.concat([x[1].iloc[get_subset_data_idx(x[1], random_seed)] for x in dg], axis=0)

    if return_arrays:
        y = array(subset_data['label'])
        subset_data.drop('label', axis=1, inplace=True)
        return subset_data.as_matrix(), y
    else:
        return subset_data
