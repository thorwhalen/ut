__author__ = 'thorwhalen'

import ut as ms
import pandas as pd
import ut.daf.get as daf_get
import ut.pstr.trans as pstr_trans
import numpy as np
import re

non_w_letter_re = re.compile('[^\w]+')


def empty_index(df):
    df.index = [''] * len(df)
    return df


def sr_set_values(sr, values):
    sr_name = sr.name
    index_names = sr.index.names
    sr = sr.reset_index()
    sr[sr_name] = values
    sr = sr.set_index(index_names)
    return sr[sr_name]


def to_lu_col_names(df, cols_to_change=None):
    cols_to_change = cols_to_change or df.columns
    new_cols = map(lambda x: x.lower(), cols_to_change)
    new_cols = map(lambda x: non_w_letter_re.sub('_', x), new_cols)
    return ch_col_names(df, new_cols, cols_to_change)


def replace_nans_with_spaces_in_object_columns(df):
    col_and_dtypes = df.dtypes.apply(str)
    for col, col_type in col_and_dtypes.iteritems():
        if col_type == 'object':
            df[col] = df[col].fillna('')
    return df


def force_col_type_as_type_of_first_element(df):
    col_types = daf_get.column_types(df)
    for k, v in col_types.iteritems():
        print "%s %s" % (k, v)
        df[k] = df[k].astype(v)
    return df


def to_utf8(df, columns=None, inplace=False):
    if inplace is False:
        df = df.copy()
    # deal with index
    processed_index = False
    index_names = df.index.names
    if index_names[0] is not None:
        processed_index = True
        df = df.reset_index()
    try:
        df.columns = pstr_trans.to_utf8_or_bust_iter(df.columns)  # change columns to utf8
    except IndexError as e:
        raise e

    if len(df) > 0:
        if columns is None:
            # default to all basestring-valued columns that are not unicode
            columns = \
                set(daf_get.column_names_whose_values_are_instances_of(df, basestring)) \
                - set(daf_get.column_names_whose_values_are_instances_of(df, str))
        for c in columns:
            df[c] = df[c].apply(pstr_trans.str_to_utf8_or_bust)
    if processed_index:
        df = df.set_index(index_names)
    if inplace is False:
        return df


def to_str(df, columns=None, inplace=False):
    if inplace is False:
        df = df.copy()
    df.columns = map(str, df.columns)  # change columns to string
    if len(df) > 0:
        if columns is None:
            # default to all basestring-valued columns that are not unicode
            columns = \
                set(daf_get.column_names_whose_values_are_instances_of(df, basestring)) \
                - set(daf_get.column_names_whose_values_are_instances_of(df, str))
        for c in columns:
            df[c] = df[c].apply(str)
    if inplace is False:
        return df


def to_unicode(df, columns=None, inplace=False):
    if inplace is False:
        df = df.copy()
    if len(df) > 0:
        if columns is None:
            # default to all basestring-valued columns that are not unicode
            columns = \
                set(daf_get.column_names_whose_values_are_instances_of(df, basestring)) \
                - set(daf_get.column_names_whose_values_are_instances_of(df, unicode))
        for c in columns:
            df[c] = df[c].apply(unicode)
    if inplace is False:
        return df


def to_unicode_with_sink(df, columns=None, inplace=False):
    if inplace is False:
        df = df.copy()
    if len(df) > 0:
        if columns is None:
            # default to all basestring-valued columns that are not unicode
            columns = \
                set(daf_get.column_names_whose_values_are_instances_of(df, basestring)) \
                - set(daf_get.column_names_whose_values_are_instances_of(df, unicode))
        for c in columns:
            df[c] = df[c].apply(_to_unicode_or_sink)
    if inplace is False:
        return df


def _to_unicode_or_sink(obj, encoding='utf-8'):
    if isinstance(obj, basestring):
        if not isinstance(obj, unicode):
            try:
                obj = unicode(obj, encoding)
            except UnicodeDecodeError:
                obj = "_" * 10
    return obj


def to_unicode_or_delete_row(df, columns=None, inplace=False):
    if inplace is False:
        df = df.copy()
    if len(df) > 0:
        if columns is None:
            # default to all basestring-valued columns that are not unicode
            columns = \
                set(daf_get.column_names_whose_values_are_instances_of(df, basestring)) \
                - set(daf_get.column_names_whose_values_are_instances_of(df, unicode))
        for c in columns:
            df[c] = df[c].apply(_to_unicode_or_nan)
        # remove all rows that contain any nan in any of the columns
        df = df.dropna(axis=0, how='any', subset=columns)
    if inplace is False:
        return df


def _to_unicode_or_nan(obj, encoding='utf-8'):
    if isinstance(obj, basestring):
        if not isinstance(obj, unicode):
            try:
                obj = unicode(obj, encoding)
            except UnicodeDecodeError:
                obj = np.nan
    return obj


def ch_col_names(df, new_names=[], old_names=None, inplace=False):
# changes the names listed in new_names to the names listed in old_names
    new_names = _force_list(new_names)
    if isinstance(df, pd.Series):
        df = df.copy()
        df.name = new_names[0]
        return df
    else:
        if old_names is None:
            old_names = list(df.columns)
        else:
            old_names = _force_list(old_names)
        assert len(new_names) == len(old_names), "old_names and new_names must be the same length"
        #return df.rename(columns={k: v for (k, v) in zip(old_names, new_names)}, inplace=inplace)
        return df.rename(columns=dict(zip(old_names, new_names)), inplace=inplace)

def ch_single_index_name(df, new_name):
    index_names = df.index.names
    if len(index_names) >= 2:
        print "you can't use ch_single_index_name if there's more than ONE index!"
        return df
    else:
        index_name = index_names[0] or 'index'
        df = df.reset_index()
        df = ch_col_names(df, new_names=[new_name], old_names=[index_name])
        df = df.set_index(new_name)
        return df


def _force_list(list_wannabe):
    if isinstance(list_wannabe, basestring):
        return [list_wannabe]
    elif not isinstance(list_wannabe, list):
        return list(list_wannabe)
    else:
        return list_wannabe