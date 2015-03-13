__author__ = 'thorwhalen'

import pandas as pd
from ut.util.ulist import all_true


def has_columns(df, cols):
    """
    returns True iff df has ALL columns listed in cols
    """
    df_cols = df.columns.tolist()
    if isinstance(cols,basestring):
        return cols in df_cols
    else:
        if not isinstance(cols,list):
            cols = cols.tolist()
        return all_true([c in df.columns.tolist() for c in cols])


def has_columns_of(df1, df2):
    """
    returns True iff df1 has ALL columns of df2
    """
    return has_columns(df1, df2.columns.tolist())


def has_single_index_named(df, index_name):
    index_names = df.index.names
    if len(index_names) != 1:
        return False
    elif index_names[0] == index_name:
        return True
    else:
        return False
