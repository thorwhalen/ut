from matplotlib.pylab import *
import pandas as pd

dflt_figsize = (16, 5)


def pplot(*args, **kwargs):
    """Long plot. Plots with a long default figsize"""
    figsize = kwargs.pop('figsize', dflt_figsize)
    if isinstance(args[0], (pd.DataFrame, pd.Series)):
        df, *args = args
        plot_func = getattr(df, 'plot')
        kwargs['figsize'] = figsize
    else:
        plot_func = plt.plot
        if figsize:
            plt.figure(figsize=figsize)
    return plot_func(*args, **kwargs)
