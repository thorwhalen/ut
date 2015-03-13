__author__ = 'thorwhalen'


#import ut.util as putil
#import re
import ut.parse.google as parse_google
from ut.semantics.term_stats_maker import TermStatsMaker
import ut.semantics.term_stats_maker as term_stats_maker
from collections import OrderedDict

LOCATION_LOCAL = 'LOCAL'
LOCATION_S3 = 'S3'

class GResultInfoRetriever(object):
    """
    GResultInfoRetriever defines a google result info parser.
    Once an instance gr_info of GResultInfo is setup, one can call gr_info.get_info(data_source)
    on data_source (html, soup, etc. of a google result's page) and get a dict holding different infos on this google result page

    NOTE: self.data_getter() (defaulted to the identity) is used to preprocess the data or even simply get the data.
    For example, we may want input html and have self.data_getter() transform it to soup, or we may want to feed the data
    as a filepath and have self.data_getter first retrieve the html and then soup it.

    NOTE: The pattern I use here is to define different parsers we want to run on the same data_source (preprocessed by
    self.data_getter()) as a dict of key,value pairs where:
        info_parsers_dict.key is the name we want to give to a piece of information
        info_parsers_dict.value is the function that should be called on data_source to get this piece of information
    The function then returns a dict of key,value pairs such that
        output.key = info_parsers_dict.key
        output.value = info_parsers_dict.value(data_source)
    """

    def __init__(self, data_getter=None, info_parsers_dict=OrderedDict()):
        self.data_getter = data_getter
        self.info_parsers_dict = info_parsers_dict

    def get_info(self, data_source):
        if self.data_getter:
            data_source = self.data_getter(data_source)
        info_dict = OrderedDict()
        for k in self.info_parsers_dict:
            info_dict[k] = self.info_parsers_dict[k](data_source)
        return info_dict

    @classmethod
    def for_nres_words_domains(cls, data_getter=None, info_parsers_dict=OrderedDict()):
        if not info_parsers_dict.has_key('number_of_results'):
            info_parsers_dict['number_of_results'] = get_number_of_results
        if not info_parsers_dict.has_key('term_stats'):
            info_parsers_dict['term_stats'] = TermStatsMaker.mk_term_stats_maker().term_stats
        if not info_parsers_dict.has_key('domain_names'):
            info_parsers_dict['domain_names'] = get_domain_term_count_from_google_results
        return GResultInfoRetriever(data_getter=data_getter, info_parsers_dict=info_parsers_dict)

    @classmethod
    def for_nres_words_domains_for_hotels(cls, data_getter=None, info_parsers_dict=OrderedDict(), location=LOCATION_LOCAL):
        if not info_parsers_dict.has_key('number_of_results'):
            info_parsers_dict['number_of_results'] = get_number_of_results
        if not info_parsers_dict.has_key('term_stats'):
            info_parsers_dict['term_stats'] = TermStatsMaker.mk_term_stats_maker_for_hotels(location=location).term_stats
        if not info_parsers_dict.has_key('domain_names'):
            info_parsers_dict['domain_names'] = get_domain_term_count_from_google_results
        return GResultInfoRetriever(data_getter=data_getter, info_parsers_dict=info_parsers_dict)



def get_number_of_results(gresults):
    if gresults.has_key('number_of_results'):
        return gresults['number_of_results']
    elif gresults.has_key('_resultStats'):
        return parse_google.parse_number_of_results(gresults['_resultStats'])

def get_domain_term_count_from_google_results(gresults):
    domain_list = []
    # if not, assume the input is a info_dict
    if gresults.has_key('organic_results_list'):
        domain_list = domain_list + [x['domain'] for x in gresults['organic_results_list'] if x.has_key('domain')]
    if gresults.has_key('top_ads_list'):
        domain_list = domain_list + [x['disp_url_domain'] for x in gresults['top_ads_list'] if x.has_key('disp_url_domain')]
    if gresults.has_key('organic_results_list'):
        domain_list = domain_list + [x['disp_url_domain'] for x in gresults['organic_results_list'] if x.has_key('disp_url_domain')]
    return term_stats_maker.list_to_term_count(domain_list)
