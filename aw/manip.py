"""
Includes functions to manipulate adwords datasets
"""
__author__ = 'thorwhalen'

import pandas as pd
import numpy as np
import re
import ut as ms

from ut.daf.check import has_columns
#import ut.util
from ut.daf.manip import lower_series
from ut.daf.manip import select_col_equals
#from ut.util.ulist import ascertain_list
import ut.pstr.trans as pstr_trans

import ut.util.ulist as util_ulist
import ut.pcoll.order_conserving as colloc
import ut.daf.manip as daf_manip
import ut.aw.reporting

from ut.util.ulist import get_first_item_contained_in_intersection_of

tokenizer_re = re.compile('[&\w]+')
non_w_re = re.compile('\W+')


def process_aw_column_names(df):
    """
    Processes (in place) column names for aw name normalization.
    It replaces some synonyms (using ut.aw.reporting.x_to_lu_name()), lower cases them, and
    replaces all non \w characters by an underscore.
    """
    df.columns = ms.aw.reporting.x_to_lu_name(list(df.columns))
    df.columns = [x.lower() for x in df.columns]
    df.columns = [non_w_re.sub('_', x) for x in df.columns]
    df.columns = ms.aw.reporting.x_to_lu_name(list(df.columns))


def process_aw_column_values(df):
    """
    Returns the df with column values in their expected format.
    (See below his py file for column_name -> processing_function map)
    """
    for col in df.columns:
        try:
            df[col] = column_names_to_preproc_function[col](df[col])
        except KeyError:
            continue
    return df


def figure_out_match_type_from_keywords_and_simplify_keywords(df):
    df = df.copy()
    if 'match_type' in df.columns:
        raise RuntimeWarning("there's a column named match_type in your df, so to be safe, I'm not going to infer it")
        return df
    elif 'keyword' not in df.columns:
        raise RuntimeError("You don't have a keyword column. I need that!")
    else:
        kw_list = np.array(df['keyword'])
        df['match_type'] = 'broad'
        for i in range(len(kw_list)):
            kw = kw_list[i]
            if kw[0] == '[':
                df['match_type'].iloc[i] = 'exact'
                df['keyword'].iloc[i] = kw[1:-1]  # remove first and last characters (should be [ and ])
            elif kw[0] == '"':
                df['match_type'].iloc[i] = 'phrase'
                df['keyword'].iloc[i] = kw[1:-1]  # remove first and last characters (should be " and ")
        return df


def mk_terms_df(df, text_cols, id_cols=None, tokenizer_re=tokenizer_re):
    text_cols = util_ulist.ascertain_list(text_cols)
    df = df.copy()
    for col in text_cols:
        df[col] = kw_str(df[col])
    return semantics_term_stats_maker_mk_terms_df(df, text_cols, id_cols, tokenizer_re)


def semantics_term_stats_maker_mk_terms_df(df, text_cols, id_cols=None, tokenizer_re=tokenizer_re):
    text_cols = util_ulist.ascertain_list(text_cols)
    if id_cols is None:
        id_cols = colloc.setdiff(df.columns, text_cols)
    else:
        id_cols = util_ulist.ascertain_list(id_cols)
        id_cols_missing = colloc.setdiff(id_cols, df.columns)
        if id_cols_missing: # if any columns are missing, try to get them from named index
            df = df.reset_index(id_cols_missing)
    dd = pd.DataFrame()
    for c in text_cols:
        d = df[id_cols]
        d['term'] = [re.findall(tokenizer_re, x) for x in df[c]]
        d = daf_manip.rollout_cols(d, cols_to_rollout='term')
        dd = pd.concat([dd, d])
    return dd


def kw_str(keyword):
    """
        produces a kw_str version of the input keyword (or list of keywords), i.e. lower ascii and strip_kw are applied
    """
    #return strip_kw(pstr_trans.lower(pstr_trans.toascii(pstr_trans.to_unicode_or_bust(keyword))))
    if isinstance(keyword, str):
        return str(
            strip_kw(
                pstr_trans.lower(
                    pstr_trans.toascii(
                        pstr_trans.to_unicode_or_bust(keyword)))))
    else:
        return [str(
            strip_kw(
                pstr_trans.lower(
                    pstr_trans.toascii(
                        pstr_trans.to_unicode_or_bust(x))))) for x in keyword]


def strip_kw(keyword):
    """
        replaces keywords (a single string or list thereof) by a version of the string that contains only characters that
        are considered by google adwords (namely, letters, numbers, and & (and underscore too, which I left hoping they'd
        never show up (they shouldn't since google doesn't allow them, or if they do, that we can just leave these)
    """
    exp = re.compile('[^\w&]', re.UNICODE)
    if isinstance(keyword, str):
        return ' '.join(re.sub(exp, ' ', keyword).split())
    else: # assume it's an iterable collection of keywords
        return [' '.join(re.sub(exp, ' ', kw).split()) for kw in keyword]

def order_words(keyword):
    """
        orders the words of a keyword string (or iterable thereof).
        NOTE: words are here defined as any string of non-space characters (you may want to consider stripping the keyword first)
    """
    #exp = re.compile('[^\w&]', re.UNICODE)
    try: # assume keyword is a string
        return ' '.join(np.sort(keyword.split(' ')))
    except: # assume it's an iterable collection of keywords
        return [' '.join(np.sort(kw.split(' '))) for kw in keyword]


def add_col(df, colname=None, overwrite=True, **kwargs):
    """
    Adds one or several requested columns (colname) to df, usually computed based on other columns of the df.
    Details of what colname does what inside the code!
    The overwrite flag (defaulted to True) specified whether
    """
    if colname is None:
        print("colname choices: ")
        print("%s" % str(['pos_impressions', 'pos', 'day_of_week_num', 'day_of_week', 'week_of_year',
                          'cvr', 'ctr', 'cpc', 'spc', 'kw_lower', 'kw_lower_ascii', 'kw_lower_ascii_ordered',
                          'destination']))
        return None
    df_columns = df.columns
    if isinstance(colname, str):
        if overwrite is False and has_columns(df, colname):
            return df # just return the df as is
        else:
            if colname == 'pos_impressions':
                df['pos_impressions'] = df['avg_position'] * df['impressions']
            elif colname == 'pos':
                df['pos'] = df['pos_impressions'] / df['impressions']
            elif colname == 'day_of_week_num':
                if 'day' in df.columns:
                    df['day_of_week_num'] = df['day'].apply(pd.datetime.weekday)
                elif 'date' in df.columns:
                    df['day_of_week_num'] = df['date'].apply(pd.datetime.weekday)
                else:
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    key_col = 'day_of_week'
                    day_2_num = pd.DataFrame({'day_of_week': days, 'day_of_week_num': np.arange(len(days))})
                    if key_col in df.index.names:
                        index_names = df.index.names
                        df = df.reset_index(drop=False)
                        df = df.merge(day_2_num)
                        if kwargs.get('rm_key_cols', False):
                            df.drop(key_col, axis=1, inplace=True)
                            index_names = list(set(index_names).difference([key_col]))
                        df = df.set_index(index_names)
                    else:
                        df = df.merge(day_2_num)
                        if kwargs.get('rm_key_cols', False):
                            df.drop(key_col, axis=1, inplace=True)
            elif colname == 'day_of_week':
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                key_col = 'day_of_week_num'
                day_2_num = pd.DataFrame({'day_of_week': days, 'day_of_week_num': np.arange(len(days))})
                if key_col in df.index.names:
                    index_names = df.index.names
                    df = df.reset_index(drop=False)
                    df = df.merge(day_2_num)
                    if kwargs.get('rm_key_cols', False):
                        df.drop(key_col, axis=1, inplace=True)
                        index_names = list(set(index_names).difference([key_col]))
                    df = df.set_index(index_names)
                else:
                    df = df.merge(day_2_num)
                    if kwargs.get('rm_key_cols', False):
                        df.drop(key_col, axis=1, inplace=True)
            elif colname == 'week_of_year':
                date_col = kwargs.get('date_col', None)
                if date_col is None:
                    date_col = get_first_item_contained_in_intersection_of(['day', 'date'], df.columns, None)
                if date_col is None:
                    raise KeyError("Couldn't find a date_col to work with: Tell me what it is")
                if isinstance(date_col, str):
                    date_col = df[date_col]
                try:
                    df['week_of_year'] = [t.isocalendar()[1] for t in date_col]
                except AttributeError:
                    df['week_of_year'] = [t.weekofyear for t in date_col]

            elif colname == 'cvr':
                df['cvr'] = df['conversions'] / df['clicks']
            elif colname == 'ctr':
                df['ctr'] = df['clicks'] / df['impressions']
            elif colname == 'cpc':
                df['cpc'] = df['cost'] / df['clicks']
            elif colname == 'spc':
                mean_cvr = sum(df['conversions']) / sum(df['clicks'])
                prior_clicks = kwargs.get('prior_clicks', 300)
                df['spc'] = (df['conversions'] + mean_cvr * prior_clicks) / (df['clicks'] + prior_clicks)
            elif colname == 'kw_lower':
                assert_dependencies(df, 'keyword', "to get {}".format(colname))
                df[colname] = lower_series(df['keyword'])
            elif colname == 'kw_lower_ascii':
                assert_dependencies(df, 'keyword', "to get {}".format(colname))
                df[colname] = pstr_trans.toascii(lower_series(df['keyword']))
            elif colname == 'kw_lower_ascii_ordered':
                assert_dependencies(df, 'keyword', "to get {}".format(colname))
                df[colname] = [' '.join(np.sort(x.split(' '))) for x in pstr_trans.toascii(lower_series(df['keyword']))]
            elif colname == 'destination':
                if 'ad_group' in df_columns:
                    #ag_triad = map(lambda x: x.split('|'), pstr_trans.lower(pstr_trans.toascii(list(df['ad_group']))))
                    ag_triad = [x.split('|') for x in df['ad_group']]
                    ag_triad_0 = kw_str([x[0] for x in ag_triad])
                    ag_triad_2 = kw_str([x[2] for x in ag_triad])
                    df[colname] = list(map(lambda x2, x0: '|'.join([x2, x0]), ag_triad_2, ag_triad_0))
                elif 'campaign' in df_columns:
                    df[colname] = kw_str(df['campaign'])
                else:
                    raise ValueError('You need ad_group or campaign to get a destination')
            else:
                raise RuntimeError("unknown colname requested")
            # remove columns?
            if 'remove_cols' in list(kwargs.keys()):
                df.drop(set(kwargs.get('remove_cols', None)).union(df.columns), axis=1, inplace=True)
    else:
        try:
            for c in colname:
                df = add_col(df, c, overwrite=overwrite)
        except TypeError:
            raise RuntimeError("colname must be a string or a list of string")
    return df


def get_kw_active(df):
    """
    returns the df with only enabled keyword status
    """
    return select_col_equals(df, 'keyword_state','enabled')


def assert_dependencies(df, cols, prefix_message=""):
    assert has_columns(df, cols), "need (all) columns {}: {}".format(util_ulist.to_str(cols), prefix_message)


# def adjust_bid(val, min_bid=0.01, max_bid=20.0, f=lambda x: x):
#     """
#     Adjusts the bids based on passed in function and ensures no bid is too low or too high.
#     Useful for operations on dataframe columns, such as when adjusting bids.
#
#     example: kw_df.max_cpc = kw_df.max_cpc.apply(adjust_bid, f=lambda x: x*2)
#     """
#     new_val = f(val)
#     new_val = min_bid if new_val < min_bid else new_val
#     new_val = max_bid if new_val > max_bid else new_val
#     return new_val

def _percentage_str_to_ratio_float(col):
    return col.apply(lambda x: x[:-1]).apply(float) / 100.

def _date_str_to_datetime(col):
    return col.apply(pd.to_datetime)

column_names_to_preproc_function = {
    'ctr': _percentage_str_to_ratio_float,
    'conv_rate': _percentage_str_to_ratio_float,
    'day': _date_str_to_datetime,
    'date': _date_str_to_datetime,
    'hour_of_day': lambda x: x.apply(int)
}

columns_that_should_be_floats = [
    'avg_position', 'impressions', 'clicks', 'conversions', 'cost_conv_', 'view_through_conv',
    'cost', 'cpc'
]

for col in columns_that_should_be_floats:
    column_names_to_preproc_function.update({col: lambda x: x.apply(float)})
