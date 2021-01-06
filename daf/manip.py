"""Dataframe manipulation tools"""
__author__ = 'thorwhalen'
"""
Includes functions to manipulate dataframes
"""
import collections
from collections import Counter
from itertools import chain

from numbers import Number

import pandas as pd
import numpy as np

from ut.daf.check import has_columns
from ut.pstr.trans import to_unicode_or_bust
import ut.util.var as util_var

import ut.util.ulist as util_ulist
import ut.pcoll.order_conserving as colloc
import ut.daf.diagnosis as daf_diagnosis
from ut.daf.ch import replace_nans_with_spaces_in_object_columns


# from ut.daf.ch import ch_col_names


def isnot_nan(x):
    try:
        return not np.isnan(x)
    except TypeError:
        return True


def none_or_type(x):
    if isnot_nan(x) and x is not None:
        return type(x)
    else:
        return None


def type_counts(df):
    return {k: dict(Counter(v)) for k, v in df.applymap(none_or_type).items()}


int_types = {int, np.int0, np.int8, np.int16, np.float64, np.float128}
float_types = {float, np.float16, np.float32, np.float64, np.float128}


# TODO: Find or make numpy type hierarchy poset (total order?) and write "maximal casting" function


def common_numerical_type(iterable):
    """
    >>> assert common_numerical_type([3.14, 1, np.nan, 2, None]) == float
    >>> assert common_numerical_type([3, 1, np.nan, 2, None]) == int
    """
    counts = set(map(none_or_type, iterable))
    if counts.isdisjoint({float, np.float16, np.float32, np.float64, np.float128}):
        return int
    else:
        return float


def determine_type(series):
    if all(isinstance(x, Number) for x in series):
        return common_numerical_type(series)
    else:
        return series.dtype


def cast_all_cols_to_numeric_if_possible(df):
    """cast to numerical all columns that have only numbers (or nan)"""
    for k, v in df.items():
        if all(isinstance(x, Number) for x in v):
            df[k] = pd.to_numeric(v)

    return df


def digitize_and_group(df, digit_cols=None, digit_agg_fun='mean', agg_fun='mean', **kwargs):
    digit_cols = digit_cols or df.columns
    t = digitize(df[digit_cols], index='int', **kwargs)
    mapping = {k: df[[k]].groupby(t[k]).agg(digit_agg_fun) for k in t.columns}
    return mapping
    # return df.groupby(list(t.to_dict(outtype='list').values())).agg(agg_fun)


def digitize(df, bins=2, index='both int and mapping', **kwargs):
    '''
    digitize(df, bins=2, index=None [, right (False)])

    df: a dataframe
    bins: number of bins or bin specification (for example, left or right (see right param) interval value)
    index: if 'int', will bins will be indexed by integers. Other options: 'int', 'by_lower_bin_val'
    '''
    kwargs['right'] = kwargs.get('right', False)
    digit_map = _mk_bins_spec(df, bins)
    d = dict()
    for c in list(digit_map.keys()):
        try:
            d[c] = np.digitize(df[c], bins=digit_map[c][1:-1], right=kwargs['right'])
        except Exception:
            continue
    # if hasattr(index, '__call__'):
    #
    if index == 'by_lower_bin_val':
        d = {c: [digit_map[c][x] for x in d[c]] for c in list(d.keys())}
        return pd.DataFrame(d, index=df.index)
    elif index == 'int':
        return pd.DataFrame(d, index=df.index)
    else:
        return pd.DataFrame(d, index=df.index), digit_map


def _mk_bins_spec(df, bins):
    if isinstance(bins, dict):
        return {c: _mk_single_variable_bin_spec(df[c], bins[c]) for c in list(bins.keys())}
    else:
        bin_spec = {c: _mk_single_variable_bin_spec(df[c], bins) for c in df.columns}
        return {c: v for c, v in bin_spec.items() if v is not None}


def _mk_single_variable_bin_spec(sr, bin_spec):
    if isinstance(bin_spec, int):
        return _unique_quantiles_bins(sr, num_of_quantiles=bin_spec + 1)
    else:
        return bin_spec


def _unique_quantiles_bins(x, num_of_quantiles=2):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    u = x.unique()
    if len(u) <= num_of_quantiles:
        return u
    else:
        p = x.quantile(q=np.linspace(0, 1, num=num_of_quantiles))
        dups = [i for i, j in list(collections.Counter(p).items()) if j > 1]
        if len(dups) > 0:
            num_of_quantiles = num_of_quantiles - len(dups)
            if num_of_quantiles >= 2:
                p = _unique_quantiles_bins(x[~x.isin(dups)], num_of_quantiles)
                return sorted(np.concatenate([p, dups]))
            else:
                RuntimeWarning("returning a few less bins than requested")
                return sorted(np.unique(p.values))
            # p = x[~x.isin(dups)].quantile(q=np.linspace(0, 1, num=max(2, num_of_quantiles - len(dups))))
            # if len(p.values) + len(dups) > num_of_quantiles:
            #     RuntimeWarning("returning a few more bins than requested")
        else:
            return p.values


def recursive_update(d, u, inplace=True):
    if inplace:
        if u:
            for k, v in list(u.items()):
                if isinstance(v, collections.Mapping):
                    recursive_update(d.get(k, {}), v, inplace=True)
                else:
                    d[k] = u[k]
    else:
        if u:
            for k, v in list(u.items()):
                if isinstance(v, collections.Mapping):
                    r = recursive_update(d.get(k, {}), v, inplace=False)
                    d[k] = r
                else:
                    d[k] = u[k]
        return d


def map_col_vals_to_ints(df, column_to_change, return_map=False):
    cols = df.columns
    unik_vals_map = ch_col_names(
        pd.DataFrame(df[column_to_change].unique()).reset_index(), ['tmp_new_col', column_to_change])
    df = pd.merge(df, unik_vals_map)
    df = rm_cols_if_present(df, column_to_change)
    df = ch_col_names(df, column_to_change, 'tmp_new_col')
    if return_map:
        return df[cols],
    else:
        return df[cols]


def hetero_concat(df_list):
    df = df_list[0]
    for dfi in df_list[1:]:
        add_nan_cols(df, colloc.setdiff(dfi.columns, df.columns))
        add_nan_cols(dfi, colloc.setdiff(df.columns, dfi.columns))
        dfi = dfi[df.columns]
        df = pd.concat([df, dfi])
    return replace_nans_with_spaces_in_object_columns(df)


def add_nan_cols(df, cols):
    for c in cols:
        df[c] = np.nan


from ut.daf.ch import ch_col_names


def rm_cols_if_present(df, cols):
    cols = util_ulist.ascertain_list(cols)
    return df[colloc.setdiff(df.columns, cols)]


def gather_col_values(df,
                      cols_to_gather=None,
                      gathered_col_name='gathered_cols',
                      keep_cols_that_were_gathered=False,
                      remove_empty_values=True):
    if cols_to_gather is None:
        cols_to_gather = df.columns
    df = df.copy()
    if remove_empty_values == False:
        df[gathered_col_name] = [list(x[1:]) for x in df[cols_to_gather].itertuples()]
    else:
        df[gathered_col_name] = \
            [[xx for xx in x if xx] for x in [list(x[1:]) for x in df[cols_to_gather].itertuples()]]
    if keep_cols_that_were_gathered == False:
        df = df[colloc.setdiff(df.columns, cols_to_gather)]
    return df


def rollin_col(df, col_to_rollin):
    """
    A sort of inverse of rollout_cols but with one column.
    Groups by those columns that are not in col_to_rollin and gathers the values of col_to_rollin in a list.
    """
    # cols = df.columns
    if isinstance(col_to_rollin, str):
        grouby_cols = list(set(df.columns) - {col_to_rollin})
    else:
        grouby_cols = list(set(df.columns) - set(col_to_rollin))
        col_to_rollin = col_to_rollin[0]

    # new code
    df_rolled = df.groupby(grouby_cols).aggregate(lambda x: tuple(x))
    df_rolled[col_to_rollin] = df_rolled[col_to_rollin].map(list)
    return df_rolled

    # # The code below stopped working (pandas changed without back compatibility)
    # try:
    #     return df.groupby(grouby_cols).agg(lambda x: [x[col_to_rollin].tolist()]).reset_index(drop=False)[cols]
    # except Exception:
    #     return df.groupby(grouby_cols).agg(lambda x: [x[col_to_rollin]])[cols] # older version


def rollout_cols(df, cols_to_rollout=None):
    """
    rolls out the values of cols_to_rollout so that each individual list (or other iterable) element is on it's own row,
    with other non-cols_to_rollout values aligned with them as in the original dataframe
    Example:
    df =
        A   B
        1   [11,111]
        2   [22]
        3   [3,33,333]
    rollout_cols(df, cols_to_rollout='B') =
        A   B
        1   11
        1   111
        2   22
        3   3
        3   33
        3   333
    """
    # if no cols_to_rollout is given, (try to) rollout all columns that are iterable (lists, etc.)
    cols_to_rollout = cols_to_rollout or daf_diagnosis.cols_that_are_of_the_type(df, util_var.is_an_iter)
    # make sure cols_to_rollout is a list
    cols_to_rollout = util_ulist.ascertain_list(cols_to_rollout)
    # get non_rollout_columns
    non_rollout_columns = colloc.setdiff(df.columns, cols_to_rollout)
    # mk an array with the lengths of the lists to rollout (get it from the first cols_to_rollout and cross fingers that
    # all cols_to_rollout have the same list lengths
    rollout_lengths = np.array(df[cols_to_rollout[0]].apply(len))
    # create a rollout_df dataframe (this will be the output)
    rollout_df = pd.DataFrame(list(range(np.sum(
        rollout_lengths))))  # TODO: I CANNOT F**ING BELIEVE I'M DOING THIS!!! But found no other way to make a dataframe empty, and then construct it on the fly!
    # rollout cols_to_rollout
    for c in cols_to_rollout:
        rollout_df[c] = np.concatenate(list(df[c]))
    # rollout cols_to_rollout
    for c in non_rollout_columns:
        t = [np.tile(x, (y, 1)) for (x, y) in zip(df[c], rollout_lengths)]
        try:
            rollout_df[c] = np.concatenate(t)
        except ValueError:
            rollout_df[c] = [x for x in chain(*t)]
    # put the columns in their original order
    return rollout_df[df.columns]


def rollout_series(d, key):
    for v in d[key]:
        dd = d.copy()
        dd[key] = v
        yield dd


def rollout_cols_alt(d, key):
    """
    Exapand
    """
    dd = list()
    for di in d.iterrows():
        dd.append(pd.DataFrame(list(rollout_series(di[1], key))))
    return pd.concat(dd)


def extract_dict_col(df, col_to_extract):
    d = [pd.DataFrame([x]) for x in df[col_to_extract]]
    return rm_cols_if_present(pd.concat([df, pd.concat(d, axis=0).reset_index(drop=True)], axis=1), col_to_extract)

    # accum_df = pd.DataFrame(columns=df.columns)
    # for i in range(len(df)):
    #     d = pd.DataFrame()


def filter_columns(df, keep_only_columns_list):
    return df[colloc.intersect(df.columns, keep_only_columns_list)]


def reorder_columns_as(df, col_order, inplace=False):
    """
    reorders columns so that they respect the order in col_order.
    Only the columns of df that are also in col_order will be reordered (and placed in front),
    those that are not will be put at the end of the returned dataframe, in their original order
    """
    if hasattr(col_order, 'columns'):
        col_order = col_order.columns
    col_order = colloc.reorder_as(list(df.columns), list(col_order))
    if not inplace:
        return df[col_order]
    else:
        col_idx_map = dict(list(zip(col_order, list(range(len(col_order))))))
        col_idx = [col_idx_map[c] for c in df.columns]
        df.columns = col_idx
        df.sort_index(axis=1, inplace=True)
        df.columns = [col_order[x] for x in df.columns]


def cols_that_are_of_the_type(df, type_spec):
    DeprecationWarning("ut.daf.manip.cols_that_are_of_the_type depreciated: "
                       "Use ut.daf.diagnosis.cols_that_are_of_the_type instead")
    if isinstance(type_spec, type):
        return [col for col in df.columns if isinstance(df[col][0], type_spec)]
    elif util_var.is_callable(type_spec):  # assume it's a boolean function, and use it as a positive filter
        return [col for col in df.columns if type_spec(df[col][0])]


def flatten_hierarchical_columns(df, sep='_'):
    """
    returns the same dataframe with hierarchical columns flattened to names obtained
    by concatinating the hierarchical names together, using sep as a seperator
    """
    df.columns = list(map(str.strip, list(map(sep.join, df.columns.values))))
    return df


def flatten_hierarchical_rows(df, sep='_'):
    """
    returns the same dataframe with hierarchical rows flattened to names obtained
    by concatinating the hierarchical names together, using sep as a seperator
    """
    df.index = list(map(str.strip, list(map(sep.join, df.index.values))))
    return df


def index_with_range(df):
    return df.reindex(index=list(range(len(df))))


def lower_series(sr):
    return sr.apply(lambda x: to_unicode_or_bust(x).lower())


def lower_column(df, col):
    df[col] = lower_series(df[col])
    return df


def select_col_equals(df, col, value):
    """
    returns the df with only enabled keyword status
    """
    assert_dependencies(df, col)
    return index_with_range(df[df[col] == value])


def rm_nan_rows(df):
    """
    removes all rows containing any nans
    """
    return index_with_range(df.dropna())


def assert_dependencies(df, cols, prefix_message=""):
    assert has_columns(df, cols), "need (all) columns {}: {}".format(util_ulist.to_str(cols), prefix_message)
