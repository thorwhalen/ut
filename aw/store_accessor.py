__author__ = 'thorwhalen'

import pfile.accessor as pfile_accessor
from analyzer.pstore import MyStore
from analyzer.pstore import StoreAccessor

local_facc = pfile_accessor.for_local('hdf5/')


def for_local(
        store_path_dict=None
):
    if store_path_dict is None:
        store_path_dict = {
            'ad_elements': local_facc('aw/ad_elements.h5'),
            'kw_elements': local_facc('aw/kw_elements.h5'),
        }
        add_from_dict = {
            'ad_group': {'join_store': 'ad_elements', 'join_cols': 'ad_group_id'}
        }
    return StoreAccessor(store_path_dict=store_path_dict, add_from_dict=add_from_dict)

def from_ad_and_kw_elements(
        store_path_dict=None
):
    if store_path_dict is None:
        store_path_dict = {
            'ad_elements': local_facc('aw/ad_elements.h5'),
            'kw_elements': local_facc('aw/kw_elements.h5'),
        }
        add_from_dict = {
            'ad_group': {'join_store': 'ad_elements', 'join_cols': 'ad_group_id'}
        }
    return StoreAccessor(store_path_dict=store_path_dict, add_from_dict=add_from_dict)


def from_stpr(
        store_path_dict=None
):
    if store_path_dict is None:
        store_path_dict = {
            'stpr': local_facc('stpr/stpr_20130107_20130507.h5')
        }
        add_from_dict = {
            'ad_group': {'join_store': 'stpr', 'join_key': 'stpr_20130107_20130507', 'join_cols': 'ad_group_id'},
            'keyword': {'join_store': 'stpr', 'join_key': 'stpr_20130107_20130507', 'join_cols': ['ad_group_id', 'search_term']}
        }
    return StoreAccessor(store_path_dict=store_path_dict, add_from_dict=add_from_dict)