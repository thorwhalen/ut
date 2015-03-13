__author__ = 'thor'

import pandas as pd
import util.ulist as util_ulist
import re
import pcoll.order_conserving as colloc


def index_aligned_inner_op(x, y, op=lambda x, y: x + y):
    xy = sr_merge(x, y)
    for c in x.columns:
        xy[c] = op(xy[c + '_x'], xy[c + '_y'])
        xy.drop([c + '_x', c + '_y'], axis=1, inplace=True)
    return xy


def cartesian_product(df1, df2):
    join_col = 'this is the joining col that wont show up elsewhere'
    df1[join_col] = 1
    df2[join_col] = 1
    df = df1.merge(df2, on=join_col)
    df1.drop(labels=join_col, axis=1, inplace=True)
    df2.drop(labels=join_col, axis=1, inplace=True)
    df.drop(labels=join_col, axis=1, inplace=True)
    return df


def sr_merge(A, B, how='inner'):
    # prepare A and B and other useful variables
    A = A.copy()
    B = B.copy()
    A_vars = A.index.names
    B_vars = B.index.names
    AB_vars_intersection = list(set(A_vars).intersection(B_vars))
    AB_vars_union = colloc.union(A_vars, B_vars)
    A = A.reset_index()
    B = B.reset_index()

    # figure out join_cols
    if len(AB_vars_intersection) == 0:  # if A and B have NO variables in common
        # remember this
        remove_join_col = True
        # add a column to join on (to have a cartesian product effect
        join_col = 'this is the joining col that wont show up elsewhere'
        A[join_col] = 1
        B[join_col] = 1
        join_cols = [join_col]
    else:
        remove_join_col = False
        join_cols = AB_vars_intersection

    AB = A.merge(B, on=join_cols, how=how, suffixes=('_x', '_y'))  # join A and B
    if remove_join_col:  # remove the cartesian join_col if necessary
        del AB[join_col]
    return AB.set_index(AB_vars_union)


def name_to_tag(name, tag_str_format='#{%s}'):
    return tag_str_format % name


def rep_tags(df, rep_cols, with_cols, name_to_tag_fun=None):
    """
    Replaces tags (specified by with_cols and the name_to_tag_fun) of the strings of rep_cols with the values of the
    with_cols columns of df.
    """
    # process inputs
    df = df.copy()
    rep_cols = util_ulist.ascertain_list(rep_cols)
    with_cols = util_ulist.ascertain_list(with_cols)
    if name_to_tag_fun is None:
        name_to_tag_fun = lambda x: name_to_tag(x, tag_str_format='#{%s}')
    tag_exp_with_col = \
        [{'with': name,
          'tag_exp': re.compile(name_to_tag_fun(name))}
         for name in with_cols]
    # go through all rep_cols and replace tags with the value in the with_cols
    for r in rep_cols:
        for t in tag_exp_with_col:
            w = t['with']
            tag_exp = t['tag_exp']
            print w, tag_exp.pattern
            df[r] = map(lambda x, y: tag_exp.sub(x, y), df[w], df[r])
    return df



