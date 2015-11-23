from __future__ import division

__author__ = 'thor'

from collections import Counter
from numpy import array


def order_clus_idx_and_centers_by_decreasing_frequencies(clus_idx, clus_centers):
    """
    Returns a version of clus_idx and clus_centers where the clus_idx (indexing into corresponding clus_centers)
    are ordered according to their frequency of occurence.

    Note: If clus_idx has gaps (i.e. doesn't have all elements from 0 to n), the missing (i.e. not indexed by
    a clus_idx) clus_centers will be removed.
    """
    clus_idx_freqs = Counter(clus_idx)  # count the clus_idx occurences
    most_common_first_clus_idx = zip(*clus_idx_freqs.most_common())[0]
    clus_idx_map = {k: v for k, v in zip(most_common_first_clus_idx, range(len(most_common_first_clus_idx)))}

    clus_idx = array(map(lambda idx: clus_idx_map[idx], clus_idx))
    clus_centers = clus_centers[array(most_common_first_clus_idx)]

    return clus_idx, clus_centers


def order_clus_idx_by_decreasing_frequencies(clus_idx):
    """
    Returns a version of clus_idx and clus_centers where the clus_idx (indexing into corresponding clus_centers)
    are ordered according to their frequency of occurence.

    Note: If clus_idx has gaps (i.e. doesn't have all elements from 0 to n), the missing (i.e. not indexed by
    a clus_idx) clus_centers will be removed.
    """
    clus_idx_freqs = Counter(clus_idx)  # count the clus_idx occurences
    most_common_first_clus_idx = zip(*clus_idx_freqs.most_common())[0]
    clus_idx_map = {k: v for k, v in zip(most_common_first_clus_idx, range(len(most_common_first_clus_idx)))}

    clus_idx = array(map(lambda idx: clus_idx_map[idx], clus_idx))

    return clus_idx
