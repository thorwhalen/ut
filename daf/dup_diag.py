"""
Includes functions to diagnose duplicates in dataframes.
Many of these functions take the following inputs:
    d: dataframe
    dup_cols: a list of columns of this dataframe (defaulting to all columns of the dataframe)
Two rows are (dup_cols-)duplicates of each other if they have exactly the same values on the dup_cols columns.
"""

import pandas as pd
from ut.util.counter import Counter
import ut.util.ulist as ulist
import numpy as np
import ut.daf.manip as daf_manip


def add_min_unique_prefix(df, token_list_col,
                           join_string=None, sep_string=None, min_unik_tok_str_col=None,
                           include_token_count=False, token_count_col=None, print_info=False, min_toks=1):
    """
    Based on a column (token_list_col) containing (1) a list of tokens or (2) a string expressing this list where tokens
    are separated by a sep_string, add_min_unique_prefix creates a column (named "unik_"+token_list_col) containing
    minimal sublists of tokens that are unique.
    For example, ['B', 'C D', 'C E F', 'A B C D', 'A B Z F'] will yield ['B', 'C D', 'C E', 'A B C', 'A B Z']
    The function returns the original dataframe with the extra "unique token strings" column.
    Note that this dataframe will not be in the same row order, and may contain less rows: The only case (I think)
    this can happen is when the original dataframe has duplicates. It is a good habit to retrieve these duplicates
    before feeding data to add_min_unique_prefix.
    """
    # preparing resources
    dup = df.copy()
    if isinstance(df[token_list_col].iloc[0], str):
        token_list_col_was_string = True
        sep_string = sep_string or join_string # use join_string as default sep_string
        dup['original_token_list_col'] = dup[token_list_col]
        dup[token_list_col] = [x.split(sep_string) for x in dup[token_list_col]]
    else:
        token_list_col_was_string = False
    join_string = join_string or sep_string or ' '
    min_unik_tok_str_col = min_unik_tok_str_col or 'mup_'+ token_list_col
    if include_token_count:
        token_count_col = token_count_col or token_list_col + '_tok_count'
    original_number_of_rows = len(dup)
    if print_info:
        print("")
        print("original number of rows: %d" % original_number_of_rows)
    dup[min_unik_tok_str_col] = ''
    dup['number_of_tokens_in_token_list'] = list(map(len, dup[token_list_col]))
    # initiate iterative process
    dup[min_unik_tok_str_col] = \
        [y[0] for y in dup[token_list_col]]
    dup, ndup_accum = dup_and_nondup_dataframes(dup, dup_cols=min_unik_tok_str_col)
    if include_token_count:
        ndup_accum[token_count_col] = 1
    if dup: # if there's still dups
        # iteratively produce unique token strings
        max_number_of_tokens = np.max(dup['number_of_tokens_in_token_list'])
        for i in range(1, max_number_of_tokens):
            if not dup:
                break # if there's no more potential dups, stop the process
            lidx = dup['number_of_tokens_in_token_list'] > i
            dup[min_unik_tok_str_col][lidx] = \
                list(map(lambda x, y : join_string.join([x, y[i]]), dup[min_unik_tok_str_col][lidx], dup[token_list_col][lidx]))
            dup, ndup = dup_and_nondup_dataframes(dup, dup_cols=min_unik_tok_str_col)
            if include_token_count:
                ndup[token_count_col] = i+1
            ndup_accum = pd.concat([ndup_accum, ndup])
            if print_info:
                print("--- phase %d ---" % i)
                print("ndup: %d" % len(ndup))
                print("dup: %d" % len(dup))
    # assuring that there's at least min_toks tokens (if possible)
    if min_toks > 1:
        ndup_accum['n_toks_in_min_unique_prefix'] = [len(x.split(join_string)) for x in ndup_accum[min_unik_tok_str_col]]
        lidx = list(map(lambda mup_n_toks, n_toks, min_toks: (mup_n_toks < min_toks) and (min_toks <= n_toks),
                   ndup_accum['n_toks_in_min_unique_prefix'],
                   ndup_accum['number_of_tokens_in_token_list'],
                   [min_toks] * len(ndup_accum)))
        ndup_accum[min_unik_tok_str_col][lidx] = \
            [join_string.join(s[:min_toks]) for s in ndup_accum[token_list_col][lidx]]
    # clean up temp columns
    ndup_accum = \
        daf_manip.rm_cols_if_present(ndup_accum, ['number_of_tokens_in_token_list', 'n_toks_in_min_unique_prefix'])
    if token_list_col_was_string:
        ndup_accum[token_list_col] = ndup_accum['original_token_list_col']
        del ndup_accum['original_token_list_col']
    # return the ndup_accum
    # if len(ndup_accum) != original_number_of_rows and print_info:
    #     print ""
    #     print "!!! Note that the unique_token_string df has %d rows (original df had %d rows" \
    #           % (len(ndup_accum), original_number_of_rows)
    #     print ""
    return ndup_accum


def ad_group_info_cols(d, grp_keys=None, grp_fun_dict={'grp_size': lambda x: len(x)}, grp_id_name='grp_id', grp_id_type='int'):
    """
    This function applies groupby(grp_keys) to the input dataframe, applies (.apply()) a list of functions
    to the groups, creating new columns to contain the results of these functions, and returns the resulting dataframe.

    ad_group_info_cols will always return the dataframe with at least one new column: the grp_id, that uniquely identifies
    the different groups (in the order they appeared in the groupby groups. By default, it will also return another column,
    the grp_size column, that specifies how many elements are in the group. The use can specify additional col_name:grp_function
    pairs in the grp_fun_dict argument

    Arguments:
        grp_id_type = 'int' (will identify groups with an integer) or 'name' will identify groups with a string or tuple
        of strings corresponding to

    Note: The grp_keys will be all columns as a default.
    Note: The grp_keys can be one or several columns, but also any keys that groupby accepts (functions, etc.)
    """
    # if grp_keys not given default to all columns
    grp_keys = grp_keys or d.columns.tolist()
    # do the groupby
    grd = d.groupby(grp_keys)
    # make the iterator for grp_id
    if grp_id_type=='name': # TODO: This doesn't work: Make it work!!!!
        my_counter = iter(grd.groups)
    elif grp_id_type=='int':
        my_counter = Counter(0)
    else:
        raise ValueError("grp_id_type must be 'int' or 'name'")
    # define the function that will be applied to every group
    if grp_id_type=='name':
        def add_grp_info(grp):
            print([ulist.ascertain_list(next(my_counter)) for i in range(len(grp))])
            grp[grp_id_name] = 0 #[ulist.ascertain_list(my_counter.next()) for i in range(len(grp))]
            for grp_fun_name,grp_fun in list(grp_fun_dict.items()):
                grp[grp_fun_name] = grp_fun(grp)
            return grp
    elif grp_id_type=='int':
        def add_grp_info(grp):
            grp[grp_id_name] = next(my_counter)
            for grp_fun_name,grp_fun in list(grp_fun_dict.items()):
                grp[grp_fun_name] = grp_fun(grp)
            return grp

    # apply add_grp_info to the groups of grd
    return grd.apply(add_grp_info)


def get_duplicates(d, dup_cols=None, keep_count=False):
    if isinstance(d, pd.Series):
        t = pd.DataFrame(d)
        tt = get_duplicates(t)
        if len(tt) > 0:
            return tt[t.columns[0]]
    else:
        dup_cols = dup_cols or d.columns.tolist()
        t = dup_and_nondup_groups(d, dup_cols)
        try:
            t = t.get_group(True)
            if not keep_count:
                del t['dup_count']
            return t
        except:
            return pd.DataFrame(columns=d.columns)


def get_non_duplicates(d, dup_cols=None, keep_count=False):
    dup_cols = dup_cols or d.columns.tolist()
    t = dup_and_nondup_groups(d,dup_cols).get_group(False)
    if not keep_count:
        del t['dup_count']
    return t


def dup_and_nondup_dataframes(d,dup_cols=None,keep_count=False):
    """
    returns a 2-tuple of dataframes D_ND = (D,ND) where
        D is a dataframe containing rows that are duplicated (dup_count>1) and
        ND is a dataframe containing rows that are not duplicated (dup_count==1)
    """
    dup_cols  = dup_cols or d.columns.tolist()
    dg = dup_and_nondup_groups(d,dup_cols)
    try:
        dup = dg.get_group(True)
    except:
        dup = pd.DataFrame(columns=d.columns)
    try:
        nondup = dg.get_group(False)
    except:
        nondup = pd.DataFrame(columns=d.columns)
    if keep_count==False:
        if len(dup)>0: del dup['dup_count']
        if len(nondup)>0: del nondup['dup_count']
    return (dup,nondup)

def dup_and_nondup_groups(d,dup_cols=None):
    """
    returns a groupby dataframe DDG where
        DDG.get_group(True) is a dataframe containing rows that are duplicated (dup_count>1) and
        DDG.get_group(False) is a dataframe containing rows that are not duplicated (dup_count==1)
    """
    dup_cols  = dup_cols or d.columns.tolist()
    dc = mk_dup_count_col(d, dup_cols)
    d = mk_dup_count_col(d, dup_cols).groupby(dc['dup_count']>1)
    return d

def mk_dup_count_col(d, dup_cols=None):
    """
    d: dataframe
    dup_cols: a list of columns of this dataframe (defaulting to all columns of the dataframe)
    mk_dup_count_col(d,dup_cols) adds a dup_count column to the dataframe d that says how many (dup_cols-)"duplicates" are in d
        More precisely, if a row r has dup_count=1 it means that this row is (dup_cols-)unique, that is, that there's only
    one row (namely r) that has exactly the same values on the dup_cols. If a row r has dup_count=3 this means that,
    besides r, there are two other rows having exactly the same values on the dup_cols.
    """
    if dup_cols is None: dup_cols = d.columns.tolist()
    # return pd.merge(d,pd.DataFrame({'dup_count':d.groupby(dup_cols).size()}),on=dup_cols)
    # index-reset version:
    return pd.merge(d, pd.DataFrame({'dup_count': d.groupby(dup_cols).size()}).reset_index(), on=dup_cols)

# def group_by_dup_count(d,dup_cols=None):
#     """
#     returns a groupby dataframe that is grouped by duplicate count (see mk_dup_count_col function)
#     """
#     if dup_cols is None: dup_cols = d.columns.tolist()
#     return mk_dup_count_col(d,dup_cols).groupby('dup_count')