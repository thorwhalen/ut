
__author__ = 'thorwhalen'
"""
Includes various adwords elements diagnosis functions
"""

#from ut.util.var import my_to_list as to_list, my_to_list
from numpy.lib import arraysetops
from numpy import array
from numpy import argmax
import pandas as pd
from ut.util.ulist import ascertain_list
import ut.util.var as util_var
from operator import eq, lt, le, gt, ge

operator_strings = dict()
operator_strings[eq] = 'are equal'
operator_strings[lt] = 'are lower than'
operator_strings[le] = 'are at most'
operator_strings[gt] = 'are greater than'
operator_strings[ge] = 'are at least'

operator_sym = dict()
operator_sym[eq] = '='
operator_sym[lt] = '<'
operator_sym[le] = '<='
operator_sym[gt] = '>'
operator_sym[ge] = '>='

def diag_df(df):
    df = df.reset_index(drop=True)  # added this 150613 because problems with obj and str indices
    cols = df.columns
    t = list()
    for c in cols:
        lidx = df[c].notnull()

        x = df[c].iloc[argmax(lidx)]

        if x == '':
            x = df[c].iloc[argmax(lidx & (array(df[c]) != ''))]
        item = {'column': c,
                'type': type(x).__name__,
                'non_null_value': x}
        try:
            item['num_uniques'] = df[c].nunique()
        except Exception:
            item['num_uniques'] = None
        try:
            item['num_nonzero'] = len(df[c].nonzero()[0])
        except Exception:
            item['num_nonzero'] = None
        try:
            item['num_nonnan'] = len(df[c].dropna())
        except Exception:
            item['num_nonnan'] = None
        t.append(item)
    return pd.DataFrame(t)


def numof(logical_series):
    return len([x for x in logical_series if x])


def pr_numof(data, column=None, op=ge, comp_val=0,
             str_format="sparse",
             op2str=None):
    """

    Examples of str_format:
        str_format="{column} {op} {comp_val}:\t {k}\t/{n} ({perc:.2f}%)" (sparse)
        str_format="{k}/{n} ({perc:.2f}%) {column} {op} {comp_val}"
        str_format="{column} that {op} {comp_val}:\t{k} (out of {n}, so {perc:.2f}%)"
        str_format="{k}/{n} ({perc:.2f}%) of {column} {op} {comp_val}"
    """
    if str_format == 'verbose':
        str_format = "{column} that {op} {comp_val}:\t{k} (out of {n}, so {perc:.2f}%)"
        op2str = op2str or operator_strings
    elif str_format == 'sparse':
        str_format = "{column} {op} {comp_val}:\t {k}\t/{n} ({perc:.2f}%)"
        op2str = op2str or operator_sym
    if op2str is None:
        op2str = operator_sym
    if column is not None:
        data = data[column]
    n = len(data)
    k = sum(op(data, comp_val))
    print((str_format.format(
        k=k,
        n=n,
        perc=100. * k / n,
        op=op2str[op],
        column=column,
        comp_val=comp_val)))


def cols_that_are_of_the_type(df, type_spec):
    if isinstance(type_spec,type):
        return [col for col in df.columns if isinstance(df[col].iloc[0], type_spec)]
    elif util_var.is_callable(type_spec): # assume it's a boolean function, and use it as a positive filter
        return [col for col in df.columns if type_spec(df[col].iloc[0])]


def get_unique(d, cols=None):
    if cols is None: cols = d.columns.tolist()
    d = d.reindex(index=list(range(len(d))))
    grouped = d.groupby(cols)
    index = [gp_keys[0] for gp_keys in list(grouped.groups.values())]
    return d.reindex(index)


def print_unique_counts(d):
    column_list = d.columns.tolist()
    print("number of rows: \t{}".format(len(d[column_list[0]])))
    print("")
    for c in column_list:
        print("number of unique {}: \t{}".format(c,len(arraysetops.unique(d[c]))))


def mk_fanout_score_df(df, fromVars, toVars, statVars=None, keep_statVars=False):
    if statVars is None:
        statVars = list(set(df.columns)-set(fromVars+toVars))
    fromVars = ascertain_list(fromVars)
    toVars = ascertain_list(toVars)
    statVars = ascertain_list(statVars)
    # make a dataframe with all same fromVars+toVars aggregated (summing the statVars)
    agg_df = df[fromVars+toVars+statVars].groupby(fromVars+toVars,as_index=False).sum()
    # group agg_df by fromVars, keeping only fromVars+statVars
    agg_df_gr = agg_df[fromVars+statVars].groupby(fromVars)
    # compute the sum-normalize values of every group
    agg_df_freq = agg_df_gr.transform(group_normalized_freq).add_suffix('_freq_fanout_ratio')
    # compute the inverse of the group sizes
    agg_df_count = agg_df_gr.agg(group_normalized_count).add_suffix('_count_fanout_ratio')
    d = agg_df.join(agg_df_freq)
    if keep_statVars==False:
        d = d.drop(statVars,axis=1)
    d = d.join(agg_df_count,on=fromVars)
    return d


def group_normalized_freq(arr):
    """
    transformation: value divided by the sum of values in the array
    """
    return arr/float(sum(arr))


def group_normalized_count(arr):
    """
    aggregation: inverse of array length
    """
    return 1.0/float(len(arr))