"""Filtering outliers"""
__author__ = 'thor'

import numpy as np


def outlier_lidx(data, method='median_dist', **kwargs):
    if method == 'median_dist':
        kwargs = dict({'thresh': 3}, **kwargs)
        thresh = kwargs['thresh']
        median_dist = np.abs(data - np.median(data))
        mdev = np.median(median_dist)
        s = median_dist/mdev if mdev else np.zeros(len(median_dist))
        return s >= thresh
    elif method == 'mean_dist':
        kwargs = dict({'thresh': 3}, **kwargs)
        thresh = kwargs['thresh']
        data_std = np.std(data)
        if data_std:
            return abs(data - np.mean(data)) / np.std(data) >= thresh
        else:
            return np.array([False for i in range(len(data))])
    else:
        raise ValueError("method not recognized")


