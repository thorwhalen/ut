__author__ = 'thorwhalen'

import pandas as pd
import ut.util.ulist as ulist
import ut.pstr.trans as pstr_trans
import ut.aw.manip as aw_manip

_Broad = 'BROAD'
_Phrase = 'PHRASE'
_Exact = 'EXACT'
_match_type_list =  [_Broad,_Phrase,_Exact]
_match_type_tag = {_Broad:'B', _Phrase:'P', _Exact:'X'}
_no_match_tag = '_'
_bpx_types = ['BPX','_PX','__X','BP_','B_X','_P_','B__']

def bid_order_stats(df):
    # count the occurances of the 7 types of BPXs
    stats = dict()
    num_of_rows = len(df)
    stats['bid_ordered'] = sum(df['bid_ordered'])
    stats['BP_bid_ordered'] = sum(df['BP_bid_ordered'])
    stats['PX_bid_ordered'] = sum(df['PX_bid_ordered'])
    stats['BX_bid_ordered'] = sum(df['BX_bid_ordered'])
    return pd.DataFrame(pd.Series(stats),columns=['count'])

def bpx_stats(df):
    # count the occurances of the 7 types of BPXs
    stats_df = pd.DataFrame(df[['bpx_tag']].groupby(['bpx_tag']).size(),columns=['count'])
    # add any BPXs that may be missing
    missing_bpx_types = list(set(_bpx_types)-set(stats_df.index))
    if len(missing_bpx_types) > 0:
        stats_df = stats_df.append(pd.DataFrame(pd.Series(data=0, index=missing_bpx_types),columns=['count']))
    stats_df.columns = ['count']
    # add a ratio column
    n_rows = len(df)
    stats_df['ratio'] = stats_df.apply(lambda x: x/float(n_rows))
    # reorder indices to the _bpx_types order
    stats_df = stats_df.reindex(_bpx_types)
    return stats_df

def add_bpx_col(df, groupby_keys=[]):
    groupby_keys = ulist.ascertain_list(groupby_keys) + ['kw_stripped_and_lowered']
    df['kw_stripped_and_lowered'] = pstr_trans.lower(aw_manip.strip_kw(df['keyword']))
    dg = df.groupby(groupby_keys,group_keys=False).apply(lambda x:_bpx_tag(x))
    del dg['kw_stripped_and_lowered']
    return dg

def _bpx_tag(grp):
    match_types = list(grp.match_type)
    bpx_tag = '' # initialize tag string
    bids_by_match = dict()
    for t in _match_type_list:
        if t in match_types:
            bpx_tag = bpx_tag + _match_type_tag[t]
            bids_by_match[t] = list(grp.max_cpc[grp.match_type==t])
        else:
            bpx_tag = bpx_tag + _no_match_tag
            bids_by_match[t] = []
    grp['bpx_tag'] = bpx_tag
    grp['BP_bid_ordered'] = True
    grp['PX_bid_ordered'] = True
    grp['BX_bid_ordered'] = True
    if bids_by_match[_Phrase]:
        grp['BP_bid_ordered'] = max(bids_by_match[_Broad]+bids_by_match[_Phrase]) == max(bids_by_match[_Phrase])
    if bids_by_match[_Exact]:
        grp['PX_bid_ordered'] = max(bids_by_match[_Phrase]+bids_by_match[_Exact]) == max(bids_by_match[_Exact])
        grp['BX_bid_ordered'] = max(bids_by_match[_Broad]+bids_by_match[_Exact]) == max(bids_by_match[_Exact])
    grp['bid_ordered'] = grp['BP_bid_ordered'] & grp['PX_bid_ordered'] & grp['BX_bid_ordered']
    return grp


def split_dolls(df):
    return df.groupby(['adgroup_id'],group_keys=False).apply(lambda x:_split_fun(x))

def _split_fun(grp):
    return pd.concat([_rep_ad_group_name(grp[grp.match_type==match],match) for match in _match_type_list])

def _rep_ad_group_name(d,match_type):
    d['ad_group'] =  [_ag_split_name(x,match_type) for x in d['ad_group']]
    return d

def _ag_split_name(ag_name,match_type):
    return ag_name + '|' + match_type






