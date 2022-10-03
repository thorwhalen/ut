__author__ = 'thor'

import collections
import pandas as pd


def counts(x):
    return pd.Series(collections.Counter(x)).sort_values(ascending=False, inplace=False)
