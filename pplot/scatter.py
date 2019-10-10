__author__ = 'thor'

import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl_plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import prettyplotlib as ppl

# Get "Set2" colors from ColorBrewer (all colorbrewer scales: http://bl.ocks.org/mbostock/5577023)

default_colors = ['#e41a1c', '#377eb8', '#4eae4b', '#994fa1', '#ff8101', '#fdfc33', '#a8572c', '#f482be', '#999999']

default_color_blind_colors = ['#b84c7d', '#59b96f', '#7f62b8', '#adb342', '#b94c3f', '#43c9b0', '#c17e36', '#738a39']

MAX_PTS_FOR_TSNE = 1500


def plot_with_label_color(X, y, shuffle=False, decompose=None, y_to_color=default_color_blind_colors, **kwargs):
    if decompose is not None:
        if decompose == True:
            if len(X) < MAX_PTS_FOR_TSNE:
                decompose = 'tsne'
            else:
                decompose = 'pca'
        if isinstance(decompose, str):
            if decompose == 'pca':
                decompose = Pipeline(steps=[('scale', StandardScaler()),
                                            ('decompose', PCA(n_components=2, whiten=True))])
            elif decompose == 'tsne':
                if len(X) > MAX_PTS_FOR_TSNE:
                    print(("TSNE would be too slow with thatm much data: Taking a set of {} random pts...".format(
                        MAX_PTS_FOR_TSNE)))
                    idx = np.random.choice(len(X), size=MAX_PTS_FOR_TSNE, replace=False)
                    X = X[idx, :]
                    y = y[idx]
                decompose = Pipeline(steps=[('scale', StandardScaler()),
                                            ('decompose', TSNE(n_components=2))])
            X = decompose.fit_transform(X)
        else:
            X = decompose(X)

    if isinstance(y[0], str):
        y = LabelEncoder().fit_transform(y)

    if len(np.unique(y)) <= len(y_to_color):
        kwargs['alpha'] = kwargs.get('alpha', 0.5)
        if shuffle:
            permi = np.random.permutation(len(X))
            X = X[permi, :]
            y = y[permi]
            for i in range(len(X)):
                mpl_plt.plot(X[i, 0], X[i, 1], 'o', color=y_to_color[y[i]], **kwargs)
        else:
            for yy in np.unique(y):
                lidx = y == yy
                mpl_plt.plot(X[lidx, 0], X[lidx, 1], 'o', color=y_to_color[yy], **kwargs)
    else:
        kwargs['alpha'] = kwargs.get('alpha', 0.4)
        mpl_plt.scatter(X[:, 0], X[:, 1], c=y, **kwargs)

    return decompose


def df_scatter_plot(df=None, x=None, y=None, label=None, **kwargs):
    if df is None:
        if y is None and x is not None:
            x, y = x[:, 0], x[:, 1]
        assert x is not None and y is not None and label is not None, "you need to specify x, y, and label"
        df = pd.DataFrame({'x': x, 'y': y, 'label': label})
        label = 'label'
        x = 'x'
        y = 'y'
    elif label is None:
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
            mpl_plt.scatter(ax, xvals, yvals, label=this_label, facecolor=colors[i], **kwargs)
        else:
            mpl_plt.scatter(ax, xvals, yvals, label=this_label, **kwargs)

    mpl_plt.legend(ax)


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

    if isinstance(factor, str):
        factor_name = factor  # save off the name
        factor = df[factor]  # extract column
        df = df.drop(factor_name, axis=1)  # remove from df, so it doesn't get a row and col in the plot.
    else:
        df = df.copy()

    classes = list(set(factor))

    if color_map is None:
        color_map = ['#e41a1c', '#377eb8', '#4eae4b', '#994fa1', '#ff8101', '#fdfc33', '#a8572c', '#f482be', '#999999']
    if not isinstance(color_map, dict):
        color_map = dict(list(zip(classes, color_map)))

    if len(classes) > len(color_map):
        raise ValueError('''Too many groups for the number of colors provided.
                            We only have {} colors in the palette, but you have {}
                            groups.'''.format(len(color_map), len(classes)))

    colors = factor.apply(lambda gr: color_map[gr])
    scatter_matrix_kwargs = dict({'figsize': (10, 10), 'marker': 'o', 'c': colors, 'diagonal': None},
                                 **kwargs.get('scatter_matrix_kwargs', {}))
    axarr = pd.tools.plotting.scatter_matrix(df, **scatter_matrix_kwargs)

    columns = list(df.columns)
    for rc in range(len(columns)):
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
