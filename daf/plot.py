__author__ = 'thor'

import numpy as np
import ut.pplot.hist
import pandas as pd
import matplotlib.pylab as plt
from ut.util.time import utc_ms_to_utc_datetime


def count_hist(sr, sort_by='value', reverse=True, horizontal=None, ratio=False, **kwargs):
    horizontal = horizontal or isinstance(sr.iloc[0], basestring)
    ut.pplot.hist.count_hist(np.array(sr), sort_by=sort_by, reverse=reverse, horizontal=horizontal, ratio=ratio,
                             **kwargs)


def col_subplots(data, legend=False, ylabels=True, figsize=None, **kwargs):
    data = pd.DataFrame(data).copy()
    if figsize is None:
        figsize = (16, min(28, 2 * len(data.columns)))

    plot_axes = data.plot(subplots=True, figsize=figsize, legend=legend, **kwargs);

    if ylabels:
        if ylabels is True:
            ylabels = data.columns
        for ax, label in zip(plot_axes, ylabels):
            ax.set_ylabel(label)

    return plot_axes


def plot_timeseries(data, time_field='index', time_type='utc_ms', legend=False, ylabels=True, figsize=None, **kwargs):
    data = pd.DataFrame(data).copy()
    if time_field in data.columns:
        data = data.set_index(time_field)
    assert time_field in data.index.names or time_field == 'index', \
        "time_field couldn't be resolved (neither in columns, nor index.name, nor 'index')"
    if time_type == 'utc_ms':
        data[time_field] = pd.to_datetime(
            np.array(map(utc_ms_to_utc_datetime, data.index.values)))
        data = data.set_index(time_field)

    if time_field == 'index':
        data.index.name = 'time'

    return col_subplots(data, legend=legend, ylabels=ylabels, figsize=figsize, **kwargs)
