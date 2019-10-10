__author__ = 'thorwhalen'
"""
Includes functions to diagnose duplicates in adwords elements
"""

from ut.aw.manip import add_col
from ut.daf.dup_diag import get_duplicates
import ut.coll.order_conserving as oc
from ut.aw.manip import assert_dependencies
import pandas as pd
import ut.daf.ch as daf_ch
import ut.daf.dup_diag as daf_dup_diag
import ut.aw.manip as aw_manip
import ut.pstr.trans as pstr_trans
import numpy as np
import collections
from ut.pstr.trans import to_unicode_or_bust

def general_stats_for_dup_diag(diag_df, n_taps=None, n_broad_taps=None, dup_types=['strip','lower','ascii']):
    if not _is_a_kw_dup_diagnosis_df(diag_df):
        # assume it is the original dataframe, so perform kw_dup_diagnosis on it (and compute n_taps and n_broad taps
        n_taps = len(diag_df)
        n_broad_taps = len(diag_df)
        diag_df = kw_dup_diagnosis(diag_df)
        return_diag_df = True
    else:
        return_diag_df = False
    # initialize stats dict
    stats = collections.OrderedDict()
    # compute the any_dup stats
    stats['number_of_kws'] = n_taps
    stats['any_dup_kw_count'] = len(diag_df)
    stats['any_dup_kw_ratio'] = len(diag_df)/float(n_taps)
    # compute the dup type stats
    for dup_type in dup_types:
        stats[dup_type+'_dup_kw_count'] = _total_dups(diag_df['dups_'+dup_type])
        stats[dup_type+'_dup_kw_ratio'] = stats[dup_type+'_dup_kw_count']/float(n_taps)
        stats[dup_type+'_dup_group_count'] = _total_dup_groups(diag_df['grp_id_'+dup_type])
    if 'dups_order' in diag_df.columns:
        dup_type = 'order'
        d = diag_df[diag_df.match_type=='Broad']
        stats[dup_type+'_dup_kw_count'] = _total_dups(d['dups_'+dup_type])
        if n_broad_taps!=0:
            stats[dup_type+'_dup_kw_ratio'] = stats[dup_type+'_dup_kw_count']/float(n_broad_taps)
        else:
            stats[dup_type+'_dup_kw_ratio'] = 0
        stats[dup_type+'_dup_group_count'] = _total_dup_groups(d['grp_id_'+dup_type])
    # return the stats dict, and also the diag_df if it wasn't given in thepowerinput
    if return_diag_df==True:
        named_tuple = collections.namedtuple('dup_stats',['stats','diag_df'])
        return named_tuple(stats=stats,diag_df=diag_df)
    else:
        return stats

def _is_a_kw_dup_diagnosis_df(df):
    if 'dups_strip' in df.columns:
        return True
    else:
        return False

def _total_dups(array_of_dup_counts):
    return len([x for x in array_of_dup_counts if x!=0])

def _total_dup_groups(array_of_dup_counts):
    return len(np.unique([x for x in array_of_dup_counts if x!=0]))

def _grp_id_array_to_number_of_groups(grp_id_array):
    unik_grp_id_array = np.unique(grp_id_array)
    nDupGroups = len(unik_grp_id_array)
    if isinstance(unik_grp_id_array[0],str):
        if '' in unik_grp_id_array:
            nDupGroups = nDupGroups - 1
    else:
        if 0 in unik_grp_id_array:
            nDupGroups = nDupGroups - 1
    return nDupGroups

def kw_dup_diagnosis(df, grp_keys=['match_type'], # grp_keys=['match_type','ad_group','campaign']?
                     grp_fun_dict={'dups': lambda x: len(x)}, grp_id_name='grp_id',grp_id_type='int',output_nondup_df=False):
    dup_df_dict = dict()
    grp_keys = oc.intersect(df.columns, grp_keys) + ['kw_representative']
    df = df.copy() # to change the input df (can be handled differently if need to spare memory)
    df.keyword = df.keyword.apply(lambda x:to_unicode_or_bust(x)) # change all keyword strings to unicode

    # util function (returns a dataframe containing grp_id and dups of a df
    def _get_grp_id_and_dups(df):
        """
        this function makes grp_id and dups duplication info columns and returns only those rows with dups>1
        NOTE: It is not meant to be used externally, but by the kw_dup_diagnosis() only
        """
        df = daf_dup_diag.ad_group_info_cols(df, grp_keys=grp_keys,
                                            grp_fun_dict=grp_fun_dict,
                                            grp_id_name=grp_id_name,
                                            grp_id_type=grp_id_type
        )
        if len(df)>0:
            return df[['grp_id', 'dups']][df.dups>1]
        else: # return an empty dataframe (but with the usual columns (necessary for the further joins)
            return pd.DataFrame(columns=['grp_id', 'dups'])

    # make a kw_representative column where different "group representatives" will be placed
    df['kw_representative'] = df['keyword']
    # get the kw_stripped duplicates
    df['kw_representative'] = aw_manip.strip_kw(df['kw_representative'])
    dup_df_dict['strip'] = _get_grp_id_and_dups(df)
    # get the kw_lower duplicates
    df['kw_representative'] = pstr_trans.lower(df['kw_representative'])
    dup_df_dict['lower'] = _get_grp_id_and_dups(df)
    # get the ascii duplicates
    df['kw_representative'] = pstr_trans.toascii(df['kw_representative'])
    dup_df_dict['ascii'] = _get_grp_id_and_dups(df)
    # get the order duplicates (only for Broads)
    d = df[df.match_type=='Broad']
    d['kw_representative'] = aw_manip.order_words(d['kw_representative'])
    dup_df_dict['order'] = _get_grp_id_and_dups(d)
    # join all this together
    d = dup_df_dict['strip'].join(dup_df_dict['lower'],how='outer',lsuffix='_strip').fillna(0)
    d = d.join(dup_df_dict['ascii'],how='outer',lsuffix='_lower').fillna(0)
    d = d.join(dup_df_dict['order'],how='outer',lsuffix='_ascii',rsuffix='_order').fillna(0)
    del df['kw_representative']
    d = d.join(df)
    if output_nondup_df==False:
        return d
    else:
        named_tuple = collections.namedtuple('dup_stats',['dup_diag_df','non_dup_df'])
        return named_tuple(dup_diag_df=d, non_dup_df=df.ix[list(set(df.index)-set(d.index))])

# def _get_grp_id_and_dups(df, grp_keys, grp_fun_dict, grp_id_name):
#     """
#     this function makes grp_id and dups duplication info columns and returns only those rows with dups>1
#     NOTE: It is not meant to be used externally, but by the kw_dup_diagnosis() only
#     """
#     df = daf_dup_diag.ad_group_info_cols(df, grp_keys=grp_keys,
#                                         grp_fun_dict=grp_fun_dict,
#                                         grp_id_name=grp_id_name
#     )
#     return df[['grp_id','dups']][df.dups>1]

def get_kw_duplicates_01(df,dup_def='kw_lower',gr_keys=['match_type','ad_group','campaign']):
    """
    old function to get kw_duplicates
    probably better to use kw_dup_diagnosis
    """
    if dup_def=='all':
        # get kw_lower dup_count df
        d = get_kw_duplicates_01(df,dup_def='kw_lower',gr_keys=gr_keys)
        d = daf_ch.ch_col_names(d,'dup_count','lower_dups')
        del d['kw_lower']
        # get kw_lower_ascii dup_count df
        dd = get_kw_duplicates_01(df,dup_def='kw_lower_ascii',gr_keys=gr_keys)
        dd = daf_ch.ch_col_names(dd,'dup_count','ascii_dups')
        del dd['kw_lower_ascii']
        # merge d and dd
        d = d.merge(dd,how='outer')
        # get kw_lower_ascii_ordered dup_count df
        dd = get_kw_duplicates_01(df,dup_def='kw_lower_ascii_ordered',gr_keys=gr_keys)
        dd = daf_ch.ch_col_names(dd,'dup_count','order_dups')
        del dd['kw_lower_ascii_ordered']
        # merge d and dd
        d = d.merge(dd,how='outer')
        # replace nans with 0s
        d = d.fillna(0)
        return d
    else:
        df = df.copy() # make a copy
        if dup_def=='kw_lower':
            d = add_col(df,'kw_lower',overwrite=False)
            gr_keys = oc.union(['kw_lower'],gr_keys)
        elif dup_def=='kw_lower_ascii':
            d = add_col(df,'kw_lower_ascii',overwrite=False)
            gr_keys = oc.union(['kw_lower_ascii'],gr_keys)
        elif dup_def=='kw_lower_ascii_ordered':
            d = df[df.match_type=='Broad']
            d = add_col(d,'kw_lower_ascii_ordered',overwrite=False)
            gr_keys = oc.union((['kw_lower_ascii_ordered'],gr_keys))
        else:
            raise ValueError("don't know how to handle that dup_def")
        assert_dependencies(d,gr_keys,"to get duplicates")
        return get_duplicates(d,gr_keys,keep_count=True)


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


# def get_duplicates(df,cols):
#     df = df.reindex(index=range(len(df)))
#     grouped = df.groupby(cols)
#     unique_index = [gp_keys[0] for gp_keys in grouped.groups.values()]
#     non_unique_index = list(set(df.index)-set(unique_index))
#     duplicates_df = get_unique(df.irow(non_unique_index),cols)
#     duplicates_df = duplicates_df[cols]
#     return df.merge(pd.DataFrame(duplicates_df))


# if __name__=="__main__":
#     print "loading some data"
#     df = pd.load('/D/Dropbox/dev/py/data/query_data/KPR--EN 01-ROW-INDIE')
#     print "get_kw_duplicates some data"
#     dd_df = get_kw_duplicates(df)
#     print "saving result"
#     df.save('/D/Dropbox/dev/py/data/query_data/dedup_KPR--EN 01-ROW-INDIE')


