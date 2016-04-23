from __future__ import division

import pandas as pd
from collections import Counter, defaultdict
from numpy import Inf
import re

__author__ = 'thor'

default_bins  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, Inf]


def multiple_degree_distributions(df, from_cols=None, to_cols=None, group_degrees=True):
    if from_cols is None:
        from_cols = df.columns
    if to_cols is None:
        to_cols = from_cols

    degdist = defaultdict(dict)
    for from_col in from_cols:
        for to_col in [x for x in to_cols if x != from_col]:
            t = degree_distribution(df, from_cols=[from_col], to_cols=[to_col], group_degrees=group_degrees)
            degdist[from_col][to_col] = t.fillna(0)

    return dict(degdist)

def degree_distribution(df, from_cols, to_cols=None, name=None, group_degrees=True):
    if isinstance(from_cols, basestring):
        from_cols = [from_cols]
    to_cols = to_cols or [x for x in df.columns if x not in from_cols]
    if isinstance(to_cols, basestring):
        to_cols = [to_cols]
    # make a name if none
    if name is None:
        name = ','.join(from_cols) + ' -< ' + ','.join(to_cols)
    # keep only relevant columns and drop duplicates
    all_cols = list(from_cols) + list(to_cols)
    df = df[all_cols].drop_duplicates()

    df = df.groupby(from_cols).size()
    df = pd.Series(Counter(df), name=name) \
        .sort_values(ascending=False, inplace=False)

    df = pd.DataFrame(df)
    if group_degrees:
        df = bin_the_counts(df)
    return df


def bin_to_bin_name(bin, last_bin="10"):
    t = str(int(re.search("\d+", bin).group()) + 1)
    if t == last_bin:
        t += "+"
    return t


# bin_names = map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]) + ["10+"]

def bin_the_counts(dd, bins=default_bins):
    dd['degree'] = pd.cut(dd.index.values, bins=bins)
    dd = dd.groupby('degree').sum()
    dd[dd.index.name] = [bin_to_bin_name(x) for x in dd.index.values]
    return dd.set_index(dd.index.name)


def plot_degree_distribution(d, logy=False):
    ax = d.plot(kind='bar', logy=logy);
    for degree, count in d.iterrows():
        degree = int(re.compile('\d+').findall(degree)[0])
        count = count.iloc[0]
        ax.text(degree - 1, count * 1.2, str(count),
                rotation=90, verticalalignment='bottom', horizontalalignment='center')