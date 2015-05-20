__author__ = 'thor'

import plotly.plotly as py
from plotly.graph_objs import *


def simple_plotly(fig):
    try:
        return py.iplot_mpl(fig)
    except Exception:
        pass

