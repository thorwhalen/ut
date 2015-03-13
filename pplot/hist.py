__author__ = 'thorwhalen'

import ut as ms
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import collections

import ut.pcoll.ordered_dict


def ihist(x):
    ux = np.unique(x)
    bins = np.array(np.sort(list(set(ux - 0.5).union(set(ux + 0.5)))))
    plt.hist(x, bins=bins)


def count_hist(x, sort_by=None, reverse=False, horizontal=False, ratio=False, **kwargs):
    kwargs = dict({'align': 'center'}, **kwargs)
    h = ms.pcoll.ordered_dict.ordered_counter(x=collections.Counter(x), sort_by=sort_by, reverse=reverse)
    if ratio:
        total = float(sum(h.values()))
        [h.__setitem__(k, v / total) for k, v in h.iteritems()]
    h_range = range(len(h))
    if horizontal:
        plt.barh(h_range, h.values(), **kwargs)
        plt.yticks(h_range, h.keys())
    else:
        plt.bar(h_range, h.values(), **kwargs)
        plt.xticks(h_range, h.keys())

