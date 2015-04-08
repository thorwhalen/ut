__author__ = 'thor'

import numpy as np
import ut.pplot.hist


def count_hist(sr, sort_by='value', reverse=True, horizontal=None, ratio=False, **kwargs):
    horizontal = horizontal or isinstance(sr.iloc[0], basestring)
    ut.pplot.hist.count_hist(np.array(sr), sort_by=sort_by, reverse=reverse, horizontal=horizontal, ratio=ratio, **kwargs)

