__author__ = 'thor'

import pandas as pd
import numpy as np
import daf.manip as daf_manip
import daf.ch as daf_ch


def smallest_unik_prefix(tok_lists, min_prefix_len=1, tok_list_col=None, list_sep=' '):
    if isinstance(tok_lists, pd.DataFrame):
        dataframe_input = True
        assert tok_list_col in tok_lists.columns, "dataframe doesn't have column: %s" % tok_list_col
        tok_lists = tok_lists.copy()
    else:
        dataframe_input = False
        tok_list_col = tok_list_col or 'tok_lists'
        tok_lists = pd.DataFrame({'tok_lists': tok_lists})

    original_cols = tok_lists.columns
    tok_lists['len_of_tok_lists'] = map(len, tok_lists[tok_list_col])
    tok_lists['list_idx'] = map(lambda x: np.min(min_prefix_len, x), tok_lists['len_of_tok_lists'])
    tok_lists['tok_str'] = map(lambda tok, idx: ' '.join(tok[:idx]))
    tok_lists['look_further'] = map(lambda x: x > min_prefix_len, tok_lists['len_of_tok_lists'])


    original_cols = tok_lists.columns
    tok_lists['tok_lists_original_order'] = range(len(tok_lists))
    tok_lists['len_of_tok_lists'] = map(len, tok_lists[tok_list_col])
    tok_lists['is_unik'] = False

    def add_is_unik():
        tok_str_count = daf_ch.ch_col_names(tok_lists[['tok_str']].groupby('tok_str').count(), 'tok_str')
    def expand_tok_prefix(idx):
        list_idx = map(lambda x: np.min(idx, x), tok_lists['len_of_tok_lists'])
        lidx = tok_lists['is_unik'] == False
        tok_lists['tok_str'][lidx] = \
            map(lambda tok, idx: list_sep.join(tok[:idx]), tok_lists[tok_list_col][lidx], list_idx[lidx])

    expand_tok_prefix(min_prefix_len)
    extra_cols_df = \
        daf_manip.rm_cols_if_present(tok_lists,
                                     set(tok_lists.columns).difference([tok_list_col, 'tok_lists_original_order']))
    max_tok_list_len = np.max(tok_lists['len_of_tok_lists'])
    work_in_progress = pd.DataFrame()
    result = pd.DataFrame()
    for i in range(min_prefix_len - 1):
        too_small_lidx = tok_lists['len_of_tok_lists'] < (i + 1)
        result = pd.concat([result, tok_lists[too_small_lidx]])


