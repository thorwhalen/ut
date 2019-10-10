__author__ = 'thorwhalen'


import ut.aw
from numpy.lib import arraysetops
import pandas as pd

import ut.daf
import ut.util


def get_unique(d,cols):
    d = d.reindex(index=list(range(len(d))))
    grouped = d.groupby(cols)
    index = [gp_keys[0] for gp_keys in list(grouped.groups.values())]
    return d.reindex(index)


def get_non_low_str_of_col(d,col):
    return d[d[col] != ut.aw.manip.lower_series(d[col])]


def print_unique_counts(d):
    column_list = d.columns.tolist()
    print("number of rows: {}".format(len(d[column_list[0]])))
    print("")
    for c in column_list:
        print("number of unique {}: {}".format(c,len(arraysetops.unique(d[c]))))


def get_kw_duplicates(df):
    df =ut.aw.manip.add_col(df,'kw_lower',overwrite=False)
    df = get_duplicates(df,'kw_lower')


def get_duplicates(df,cols):
    df = df.reindex(index=list(range(len(df))))
    grouped = df.groupby(cols)
    unique_index = [gp_keys[0] for gp_keys in list(grouped.groups.values())]
    non_unique_index = list(set(df.index)-set(unique_index))
    duplicates_df = ut.aw.diagnosis.get_unique(df.irow(non_unique_index),cols)
    duplicates_df = duplicates_df[cols]
    return df.merge(pd.DataFrame(duplicates_df))


def add_col(df,colname,overwrite=True):
    """
    Adds one or several requested columns (colname) to df, usually computed based on other columns of the df.
    Details of what colname does what inside the code!
    The overwrite flag (defaulted to True) specified whether
    """
    if isinstance(colname,str):
        if overwrite==False and ut.daf.check.has_columns(df,colname):
            return df # just return the df as is
        else:
            if colname=='lower_kw':
                assert_dependencies(df,'Keyword',"to get {}".format(colname))
                df['lower_kw'] = lower_series(df['Keyword'])
            else:
                raise RuntimeError("unknown colname requested")
    elif isinstance(colname,list):
        for c in colname:
            df = add_col(df,c,overwrite=overwrite)
    else:
        raise RuntimeError("colname must be a string or a list of string")
    return df

def index_with_range(df):
    return df.reindex(index=list(range(len(df))))

def lower_series(sr):
    return sr.apply(lambda x: x.lower())

def lower_column(df,col):
    df[col] = lower_series(df[col])
    return df

def get_kw_active(df):
    """
    returns the df with only enabled keyword status
    """
    return select_col_equals(df,'Keyword state','enabled')

def select_col_equals(df,col,value):
    """
    returns the df with only enabled keyword status
    """
    assert_dependencies(df,col)
    return index_with_range(df[df[col]==value])

def rm_nan_rows(df):
    """
    removes all rows containing any nans
    """
    return index_with_range(df.dropna())


def assert_dependencies(df,cols,prefix_message=""):
    assert ut.daf.check.has_columns(df,cols),"need (all) columns {}: {}".format(ut.util.ulist.to_str(cols),prefix_message)