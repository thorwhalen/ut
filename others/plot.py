__author__ = 'thor'


def factor_scatter_matrix(df, factor, palette=None, **kwargs):
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
    import numpy as np
    from pandas.tools.plotting import scatter_matrix
    from scipy.stats import gaussian_kde
    # from pyqt_fit import kde

    if isinstance(factor, str):
        factor_name = factor  # save off the name
        factor = df[factor]  # extract column
        df = df.drop(factor_name, axis=1)  # remove from df, so it doesn't get a row and col in the plot.

    classes = list(set(factor))

    if palette is None:
        palette = ['#e41a1c', '#377eb8', '#4eae4b',
                   '#994fa1', '#ff8101', '#fdfc33',
                   '#a8572c', '#f482be', '#999999']

    color_map = dict(list(zip(classes, palette)))

    if len(classes) > len(palette):
        raise ValueError('''Too many groups for the number of colors provided.
We only have {} colors in the palette, but you have {}
groups.'''.format(len(palette), len(classes)))

    colors = factor.apply(lambda gr: color_map[gr])
    scatter_matrix_kwargs = dict({'figsize': (10, 10), 'marker': 'o', 'c': colors, 'diagonal': None},
                                 **kwargs.get('scatter_matrix_kwargs', {}))
    axarr = scatter_matrix(df, **scatter_matrix_kwargs)

    columns = list(df.columns)
    for rc in range(len(columns)):
        for group in classes:
            y = df[factor == group].icol(rc).values
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
