__author__ = 'thor'

import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl_plt
import prettyplotlib as ppl

# Get "Set2" colors from ColorBrewer (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)

default_colors = ['#e41a1c', '#377eb8', '#4eae4b', '#994fa1', '#ff8101', '#fdfc33', '#a8572c', '#f482be', '#999999']


def df_scatter_plot(df, x=None, y=None, label=None, **kwargs):

    if label is None:
        if len(df.columns) != 3:
            raise ValueError("I can't (or rather won't) guess the label if there's not exactly 3 columns. "
                             "You need to specify it")
        else:
            label = [t for t in df.columns if t not in [x, y]][0]
    colors = kwargs.pop('colors', None)
    label_list = kwargs.pop('label_list', np.array(df[label].unique()))
    fig, ax = mpl_plt.subplots(1)

    for i, this_label in enumerate(label_list):
        d = df[df[label] == this_label]
        xvals = np.array(d[x])
        yvals = np.array(d[y])
        if colors:
            ppl.scatter(ax, xvals, yvals, label=this_label, facecolor=colors[i], **kwargs)
        else:
            ppl.scatter(ax, xvals, yvals, label=this_label, **kwargs)

    ppl.legend(ax)


def factor_scatter_matrix(df, factor, color_map=None, **kwargs):
    '''Create a scatter matrix of the variables in df, with differently colored
    points depending on the value of df[factor].
    inputs:
        df: pandas.DataFrame containing the columns to be plotted, as well
            as factor.
        factor: string or pandas.Series. The column indicating which group
            each row belongs to.
        palette: A list of hex codes, at least as long as the number of groups.
            If omitted, a predefined palette will be used, but it only includes
            9 groups.
    '''
    # import matplotlib.colors
    from scipy.stats import gaussian_kde
    # from pyqt_fit import kde

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    if isinstance(factor, np.ndarray):
        factor = pd.Series(factor)

    if isinstance(factor, basestring):
        factor_name = factor  # save off the name
        factor = df[factor]  # extract column
        df = df.drop(factor_name, axis=1)  # remove from df, so it doesn't get a row and col in the plot.
    else:
        df = df.copy()

    classes = list(set(factor))

    if color_map is None:
        color_map = ['#e41a1c', '#377eb8', '#4eae4b', '#994fa1', '#ff8101', '#fdfc33', '#a8572c', '#f482be', '#999999']
    if not isinstance(color_map, dict):
        color_map = dict(zip(classes, color_map))

    if len(classes) > len(color_map):
        raise ValueError('''Too many groups for the number of colors provided.
                            We only have {} colors in the palette, but you have {}
                            groups.'''.format(len(color_map), len(classes)))

    colors = factor.apply(lambda gr: color_map[gr])
    scatter_matrix_kwargs = dict({'figsize': (10, 10), 'marker': 'o', 'c': colors, 'diagonal': None},
                                 **kwargs.get('scatter_matrix_kwargs', {}))
    axarr = pd.tools.plotting.scatter_matrix(df, **scatter_matrix_kwargs)

    columns = list(df.columns)
    for rc in xrange(len(columns)):
        for group in classes:
            y = df[factor == group].iloc[:, rc].values
            gkde = gaussian_kde(y)
            ind = np.linspace(y.min(), y.max(), 1000)
            # if columns[rc] in log_axes:
            #     est = kde.KDE1D(ind, method='linear_combination', lower=0)
            #     kde_ind = kde.TransformKDE(est, kde.LogTransform)
            # else:
            #     kde_ind = gkde.evaluate(ind)
            axarr[rc][rc].plot(ind, gkde.evaluate(ind), c=color_map[group])

    # for r in xrange(len(columns)):
    #     for c in xrange(len(columns)):
    #
    #         a = axarr[r][c]
            # if columns[r] in log_axes:
            #     # print "%d,%d: %s" % columns[r]
            #     a.set_yscale('symlog')
            # if columns[c] in log_axes:
            #     # print "%d,%d: %s" % columns[c]
            #     a.set_xscale('symlog')

    return axarr, color_map
