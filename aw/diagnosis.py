__author__ = 'thorwhalen'
"""
Includes various adwords elements diagnosis functions
"""

#from daf.manip import lower_series

# from numpy.lib import arraysetops
# import pandas as pd

def ad_group_ids_are_unique(df):
    """
    This function returns True iff ad_group_ids are unique (only show up once in the rows of df)
    """
    return len(df[['ad_group_id']].drop_duplicates()) == len(df)

def ad_group_id_does_not_need_campaign_id(df):
    """
    This function returns True iff ad_group_ids are unique globally (that is, without the need of campaign_id)
    """
    return len(df[['ad_group_id']].drop_duplicates()) == len(df[['ad_group_id', 'campaign_id']].drop_duplicates())

def keyword_ids_are_unique(df):
    """
    This function returns True iff keyword_ids are unique (only show up once in the rows of df)
    """
    return len(df[['keyword_id']].drop_duplicates()) == len(df)

def keyword_id_does_not_need_ad_group_and_campaign_ids(df):
    """
    This function returns True iff keyword_ids are unique globally (that is, without the need of ag and cp ids)
    """
    return len(df[['keyword_id']].drop_duplicates())\
           == len(df[['keyword_id', 'ad_group_id', 'campaign_id']].drop_duplicates())

# def get_non_low_str_of_col(d,col):
#     return d[d[col]!=lower_series(d[col])]



