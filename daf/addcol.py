__author__ = 'thor'


import ut.util.ulist as ulist
import ut.daf.get
import ut.daf.gr
from ut.parse.web.url import get_domain_and_suffix
from ut.parse.web.url import get_sub_domain_and_suffix
import tldextract


def group_count(df, gr_cols=None, count_col=None, keep_order=True):
    """
    adds a column containing the count of the number of groups (defined by the gr_cols columns)
    """
    gr_cols = gr_cols or df.columns
    count_col = count_col or ut.daf.get.free_col_name(df, ['count', 'gr_count'])
    if keep_order:
        df = df.copy()
        df['column_to_keep_original_order'] = list(range(len(df)))
    gr_cols = ulist.ascertain_list(gr_cols)
    gr_df = ut.daf.gr.group_and_count(df[gr_cols], count_col=count_col)
    df = df.merge(gr_df, left_on=gr_cols, right_on=gr_cols)
    if keep_order:
        df.sort(columns=['column_to_keep_original_order'], inplace=True)
        del df['column_to_keep_original_order']
    return df


def has_duplicate(df, gr_cols=None, has_dup_col=None, keep_order=True):
    """
    Adds a column indicating if the row has a duplicate (defined by equality over gr_cols columns.
    """
    count_col = 'col_revealing_how_many_times_a_group_instance_shows_up'
    gr_cols = gr_cols or df.columns
    df = group_count(df, gr_cols=gr_cols, count_col=count_col, keep_order=keep_order)
    has_dup_col = has_dup_col or 'has_duplicate'
    df[has_dup_col] = df[count_col] > 1
    del df[count_col]
    return df


def url_registered_domain(df, url_col='url', new_col='registered_domain'):
    df[new_col] = df[url_col].map(lambda x: tldextract.extract(x).registered_domain)
    return df


def url_domain_and_suffix(df, url_col='url', new_col='domain_and_suffix'):
    df[new_col] = df[url_col].map(get_domain_and_suffix)
    return df


def url_subdomain_and_suffix(df, url_col='url', new_col='subdomain_and_suffix'):
    df[new_col] = df[url_col].map(get_sub_domain_and_suffix)
    return df
