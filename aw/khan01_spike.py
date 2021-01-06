"""Travel adwords tools"""
__author__ = 'thorwhalen'

import numpy as np
from ut.datapath import datapath
import ut.daf.diagnosis as daf_diagnosis
import ut.parse.google as google
import ut.parse.util as parse_util
import pandas as pd
import ut.daf.ch as daf_ch

import os
import re
from ut.pstr.trans import toascii
from . import reporting
from datetime import datetime
from ut.util.ulist import ascertain_list
from ut.util.ulist import all_true
from ut.pstr.trans import to_unicode_or_bust
from serialize.data_accessor import DataAccessor

split_exp = re.compile("[^&\w]*")

travel_domain_list = ['expedia', 'tripadvisor', 'trivago', 'marriott', 'booking', 'hotels', 'lastminute', 'accorhotels',
                      'kayak', 'venere', 'hilton', 'hotelscombined', 'agoda', 'choicehotels', 'travelocity',
                      'travelsupermarket', 'bestwestern', 'laterooms', 'radissonblu', 'hotwire', 'lonelyplanet',
                      'orbitz', 'starwoodhotels', 'frommers', 'hotel', 'hotelclub', 'hrs', 'novotel', 'wego', 'wotif',
                      'hoteltravel', 'hyatt', 'ibis', 'ihg', 'mercure', 'priceline', 'qualityinn',
                      'beoo', 'easytobook', 'ebookers', 'hostelbookers', 'lq', 'melia', 'millenniumhotels', 'mrandmrssmith',
                      'nh-hotels', 'ratestogo', 'sofitel', 'tablethotels', 'travelandleisure']

html_data = DataAccessor('html/google_results_tests')

def get_search_term_html(search_term):
    file_name = html_data+'.html'
    try:
        return html_data.loads(file_name)
    except:
        IOError("didn't find %s" % html_data.filepath(file_name))

def google_light_parse(gresult):
    gresult = parse_util.x_to_soup(gresult)
    parse_dict = dict()
    resultStats = input.find(name='div',attrs={'id':'resultStats'})
    if resultStats:
        parse_dict['_resultStats'] = google.parse_number_of_results(resultStats)




def save_search_term_that_does_not_have_num_of_results(search_term):
    print("no num_of_results in: %s" % search_term)

def add_travel_score(query_report_df, html_folder=datapath(),ext='.html'):
    pass


# TODO: Continue coding add_travel_score()
def mk_search_term_domains_df(query_report_df, html_folder=datapath(),ext='.html'):
    search_terms = np.unique(list(query_report_df['search_term']))
    domain_lists = []
    search_term_list = []
    for st in search_terms:
        filename = os.path.join(html_folder,st+ext)
        print(filename)
        if os.path.exists(filename):
            search_term_list.append(st)
            domain_lists.append(get_domain_list_from_google_results(filename))
    return pd.DataFrame({'search_term':search_term_list, 'domain_list':domain_lists})

def add_search_term_ndups(df, count_var='ndups'):
    d = df[['search_term']].groupby('search_term').count()
    d.columns = [count_var]
    return df.merge(d, left_on='search_term', right_index=True)

def add_target_scores(query_report_df):
    vars_to_keep = ['search_term','impressions','destination','ad_group','destination_imps_freq_fanout_ratio','ad_group_imps_freq_fanout_ratio']
    query_report_df = add_query_fanout_scores(query_report_df)
    query_report_df = query_report_df[vars_to_keep]
    query_report_df = daf_ch.ch_col_names(query_report_df,
                                   ['ad_group_score','destination_score'],
                                   ['ad_group_imps_freq_fanout_ratio','destination_imps_freq_fanout_ratio'])
    query_report_df = query_report_df.sort(columns=['search_term','destination_score','ad_group_score'])
    return query_report_df

def add_query_fanout_scores(query_report_df):
    if 'destination' not in query_report_df.columns:
        query_report_df = add_destination(query_report_df)
    ad_group_fanout = mk_query_fanout_scores(query_report_df,target='ad_group',statVars='impressions',keep_statVars=False)
    ad_group_fanout = daf_ch.ch_col_names(ad_group_fanout,
                                   ['ad_group_imps_freq_fanout_ratio','ad_group_count_fanout_ratio'],
                                   ['impressions_freq_fanout_ratio','impressions_count_fanout_ratio'])
    destination_fanout = mk_query_fanout_scores(query_report_df,target='destination',statVars='impressions',keep_statVars=False)
    destination_fanout = daf_ch.ch_col_names(destination_fanout,
                                   ['destination_imps_freq_fanout_ratio','destination_count_fanout_ratio'],
                                   ['impressions_freq_fanout_ratio','impressions_count_fanout_ratio'])
    query_report_df = query_report_df.merge(ad_group_fanout,on=['search_term','ad_group'])
    query_report_df = query_report_df.merge(destination_fanout,on=['search_term','destination'])
    return query_report_df


def mk_query_fanout_scores(query_report_df, target='ad_group', statVars='impressions',keep_statVars=False):
    # target = ascertain_list(target)
    # if ('destination' in target) and ('destination' not in query_report_df.columns):
    #     query_report_df['destination'] = query_report_df['ad_group'].apply(lambda x : re.match('[^|]*',x).group(0))
    # if not all_true([x in query_report_df.columns for x in target]):
    #     raise ValueError("the dataframe doesn't have the column %s and I don't know how to make it" % target)
    if target not in query_report_df.columns:
        if target=='destination':
            query_report_df = add_destination(query_report_df)
        else:
            raise ValueError("the dataframe doesn't have the column %s and I don't know how to make it" % target)
    return daf_diagnosis.mk_fanout_score_df(query_report_df,fromVars=['search_term'],toVars=target,statVars=statVars,keep_statVars=keep_statVars)

def add_destination(query_report_df):
    assert 'ad_group' in query_report_df.columns, "you must have the variable ad_group to infer destination"
    # get the destination as the substring before the first |
    query_report_df['destination'] = query_report_df['destination'] = \
        query_report_df['ad_group'].apply(lambda x : re.match('[^|]*',x).group(0))
    return query_report_df

def get_domain_list_from_google_results(gresults):
    domain_list = []
    if not isinstance(gresults,dict): # assume it's a soup, html, or filename thereof
        gresults = google.parse_tag_dict(google.mk_gresult_tag_dict(gresults))
    # if not, assume the input is a info_dict
    if 'organic_results_list' in gresults:
        domain_list = domain_list + [x['domain'] for x in gresults['organic_results_list'] if 'domain' in x]
    if 'top_ads_list' in gresults:
        domain_list = domain_list + [x['disp_url_domain'] for x in gresults['top_ads_list'] if 'disp_url_domain' in x]
    if 'organic_results_list' in gresults:
        domain_list = domain_list + [x['disp_url_domain'] for x in gresults['organic_results_list'] if 'disp_url_domain' in x]
    return domain_list


def similarity_with_travel_domains(domain_list):
    if isinstance(domain_list, pd.DataFrame):
        domain_list = list(domain_list.index)
    return len(set(domain_list).intersection(set(travel_domain_list)))\
               / float(len(domain_list))



###############################################################################################
## SEMANTIC STUFF


def mk_term_count_from_google_results(gresults):
    """
    takes a google result (in the form of html, filename thereof, soup, or info_dict, and
    returns a Series whose indices are terms and values are term counts
    (the number of times the term appeared in the google result)
    """
    # get preprocessed text from gresults
    gresults = process_text_for_word_count(mk_text_from_google_results(gresults))
    # tokenize this text
    toks = tokenize_text(gresults)
    # make a dataframe of term counts TODO: Explore faster ways to do this
    df = pd.DataFrame(toks,columns=['token'])
    df = df.groupby('token').count()
    df.columns = ['count']
    df = df.sort(columns=['count'],ascending=False) # TODO: Take out sorting at some point since it's unecessary (just for diagnosis purposes)
    return df

def tokenize_text(gresult_text):
    return re.split(split_exp,gresult_text)

def process_text_for_word_count(text):
    """
    Preprocesses the text before it will be fed to the tokenizer.
    Here, we should put things like lower-casing the text, casting letters to "simple" ("ascii", "non-accentuated")
    letters, replacing some common strings (such as "bed and breakfast", "New York" by singular token representatives
    such as "b&b", "new_york"), and what ever needs to be done before tokens are retrieved from text.
    """
    return toascii(to_unicode_or_bust(text)).lower()


def mk_text_from_google_results(gresults):
    if not isinstance(gresults,dict): # if not a dict assume it's a soup, html, or filename thereof
        gresults = google.parse_tag_dict(google.mk_gresult_tag_dict(gresults))
    if 'organic_results_list' in gresults:
        title_text_concatinated = ' '.join([x['title_text'] for x in gresults['organic_results_list'] if 'title_text' in x])
        snippet_text_concatinated = ' '.join([x['st_text'] for x in gresults['organic_results_list'] if 'st_text' in x])
        text_concatinated = title_text_concatinated + ' ' + snippet_text_concatinated
    else:
        search_for_tag = ['_ires','_search','_res','_center_col']
        for t in search_for_tag:
            if t in gresults:
                text_concatinated = soup_to_text(gresults[t])
                break
        if not text_concatinated: # if you still don't have anything
            text_concatinated = soup_to_text(gresults) #... just get the text from the whole soup
    return text_concatinated

def soup_to_text(element):
    return list(filter(visible, element.findAll(text=True)))

def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element)):
        return False
    return True



###############################################################################################
## MAKING DATA

def mk_df_of_travel_domains():
    # set up resources
    html_folder = '/D/Dropbox/dev/py/data/html/google_results_tests/'
    file_list = ['hotel - 100 Google Search Results.html',
             'find hotel deals - 100 Google Search Results.html',
             'hotel travel sites - 100 Google Search Results.html',
             'find hotels - 100 Google Search Results.html',
             'hotel paris - 100 Google Search Results.html',
             'hotel rome - 100 Google Search Results.html',
             'hotel london - 100 Google Search Results.html',
             'hotel nyc - 100 Google Search Results.html',
             'hotels in france - 100 Google Search Results.html',
             'hotels in italy - 100 Google Search Results.html'
            ]
    filepath_list = [os.path.join(html_folder,f) for f in file_list]
    # parse all this
    r = [google.mk_gresult_tag_dict(f) for f in filepath_list]
    r = [google.parse_tag_dict(f) for f in r]
    # make domain lists
    org_domain_list = []
    ads_domain_list = []
    tads_domain_list = []
    for rr in r:
        rrr = rr['organic_results_list']
        org_domain_list = org_domain_list + [x['domain'] for x in rrr if 'domain' in x]
        rrr = rr['rhs_ads_list']
        ads_domain_list = ads_domain_list + [x['disp_url_domain'] for x in rrr if 'disp_url_domain' in x]
        rrr = rr['top_ads_list']
        ads_domain_list = ads_domain_list + [x['disp_url_domain'] for x in rrr if 'disp_url_domain' in x]
    domain_list = org_domain_list + ads_domain_list
    print("number of org_domain_list entries = %d" % len(org_domain_list))
    print("number of ads_domain_list entries = %d" % len(ads_domain_list))
    print("number of (all) domain_list entries = %d" % len(domain_list))
    # make a dataframe counting the number of times we encouter each domain
    df = pd.DataFrame(domain_list,columns=['domain'])
    dg = df.groupby('domain').count() #agg([('domain_count','len')])
    dg = daf_ch.ch_col_names(dg,'count','domain')
    thresh = 4
    print("length before removing count<%d entries = %d" % (thresh,len(dg)))
    dg = dg[dg['count']>=thresh]
    print("length before removing count<%d entries = %d" % (thresh,len(dg)))
    dg['frequency'] = dg['count']/float(max(dg['count']))
    dg = dg.sort(columns=['count'],ascending=False)
    dg.head(30)
    # return this!
    return dg


