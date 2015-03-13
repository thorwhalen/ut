__author__ = 'thor'

import ut as ms
import ut.util.ulist
import ut.daf.ch
import ut.daf.get
import pandas as pd


def group_and_count(df, count_col=None, frequency=False):
    if isinstance(df, pd.Series):
        t = pd.DataFrame()
        t[df.name] = df
        df = t
        del t
    count_col = count_col or ms.daf.get.free_col_name(df, ['count', 'gr_count'])
    # gr_cols = list(df.columns)
    # d = ms.daf.ch.ch_col_names(df.groupby(gr_cols).count()[[gr_cols[0]]], count_col).reset_index()
    d = df.copy()
    d[count_col] = 1
    d = d.groupby(list(df.columns)).count().reset_index()
    if frequency:
        d[count_col] /= float(d[count_col].sum())
    return d


def group_and_gather_unique_values_of_cols(df, groupby_cols, gather_col=None):
    # input processing
    groupby_cols = ms.util.ulist.ascertain_list(groupby_cols)
    if gather_col is None:
        assert len(df.columns) == (len(groupby_cols)+1), \
            "I can't guess what the gather_col is in your case (you must have exactly len(groupby_cols)+1 columns"
        gather_col = (set(df.columns) - set(groupby_cols)).pop()
    df = df[groupby_cols + [gather_col]]
    # the actual thing the function should do:
    return df.groupby(groupby_cols).agg(lambda x: [list(x[gather_col].unique())])


def group_and_gather_values(df, groupby_cols, gather_col=None):
    # input processing
    groupby_cols = ms.util.ulist.ascertain_list(groupby_cols)
    if gather_col is None:
        assert len(df.columns) == (len(groupby_cols)+1), \
            "I can't guess what the gather_col is in your case (you must have exactly len(groupby_cols)+1 columns"
        gather_col = (set(df.columns) - set(groupby_cols)).pop()
    df = df[groupby_cols + [gather_col]]
    # the actual thing the function should do:
    return df.groupby(groupby_cols).agg(lambda x: [list(x[gather_col])])

# def group_by_non_hashable_columns(df, groupby_cols, **kwargs):
#     if isinstance(groupby_cols, basestring):
#         groupby_cols = [groupby_cols]
#     groupby_cols_original_types = {c: type(df[c].iloc[0]) for c in groupby_cols}
#     df[groupby_cols] = df[groupby_cols].map(tuple)
#     dg = df.groupby(map(tuple, df[groupby_cols]))
#     df = df.reset_index(drop=False)
#     for c, ctype in groupby_cols_original_types.iteritems():
#         df[c] = map(ctype, df[c])