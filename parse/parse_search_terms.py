"""Parsing search terms"""
__author__ = 'thorwhalen'

import functools

import ut.util.data_source as util_data_source
from serialize.data_accessor import DataAccessor
#from ut.pfile.accessor import Accessor
import ut.parse.google as parse_google
#import ut.aw.khan01_spike as khan_spike
#from ut.semantics.term_stats_maker import TermStatsMaker
import ut.parse.gresult_info_retriever as parse_gresult_info_retriever
#import numpy as np
import pandas as pd
from serialize.khan_logger import KhanLogger
#import ut.datapath as datapath
#from serialize.s3 import S3
# import os
import ut.pfile.accessor as pfile_accessor
import logging
from khan_utils.encoding import to_utf8_or_bust



class ParseSearchTerms(object):

    # def __init__(self,
    #              get_and_save_html_if_not_saved_already=None,
    #              parse=None,
    #              diagnose_parsed_result=None,
    #              do_something_with_good_parsed_result=None,
    #              get_info=None,
    #              diagnose_info_dict=None,
    #              do_something_with_good_info_dict=None,
    #              log_failure=None
    #              ):
    #     self.get_and_save_html_if_not_saved_already = get_and_save_html_if_not_saved_already
    #     self.parse = parse
    #     self.diagnose_parsed_result = diagnose_parsed_result
    #     self.do_something_with_good_parsed_result = do_something_with_good_parsed_result
    #     self.get_info = get_info
    #     self.diagnose_info_dict = diagnose_info_dict
    #     self.do_something_with_good_info_dict = do_something_with_good_info_dict
    #     self.log_failure = log_failure

    def process(self, search_term):
        try:
            try: # getting the html for this search_term
                html = self.get_html(search_term)
            except Exception as e:
                self.logger.warn(msg='Could not get html for {}. '.format(to_utf8_or_bust(search_term)) + e.message)
                return

            try: # parsing the html for this search_term
                parsed_result = self.parse(html)
                try: # diagnose the parsed_result to decide where to go from here
                    diagnosis_string = self.diagnose_parsed_result(parsed_result)
                    if diagnosis_string: # non-empty diagnosis_string means diagnosis failed
                        self.logger.warn(msg= "Diagnosis failed for {} : {}".format(search_term, diagnosis_string))
                    else: # empty diagnosis_string means diagnosis succeeded
                        self.do_something_with_good_parsed_result(search_term, parsed_result)
                except Exception as e:
                    self.logger.warn(msg="diagnose_parsed_result_exception for {} : {}".format(to_utf8_or_bust(search_term), e.message))
                    return
            except Exception as e:
                self.logger.warn(msg="parse exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                return

            try: # getting info dict from parsed_result
                info_dict = self.get_info(parsed_result)
                try: # diagnose the info_dict to decide where to go from here
                    diagnosis_string = self.diagnose_info_dict(info_dict)
                    if diagnosis_string: # non-empty diagnosis_string means diagnosis failed
                        self.logger.warn(msg= "Diagnosis failed for {} : {}".format(search_term, diagnosis_string))
                    else: # empty diagnosis_string means diagnosis succeeded
                        self.do_something_with_good_info_dict(search_term, info_dict)
                except Exception as e:
                    self.logger.warn(msg="diagnose_info_dict_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                    return
            except Exception as e:
                self.logger.warn(msg="get_info_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                return

        except Exception as e:
            self.logger.error(msg="untracked_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))


    def html_to_parsed_result(self, search_term, html):
        try: # parsing the html for this search_term
            parsed_result = self.parse(html)
            try: # diagnose the parsed_result to decide where to go from here
                diagnosis_string = self.diagnose_parsed_result(parsed_result)
                if diagnosis_string: # non-empty diagnosis_string means diagnosis failed
                    self.logger.warn(msg= "Diagnosis failed for {} : {}".format(search_term, diagnosis_string))
                else: # empty diagnosis_string means diagnosis succeeded
                    self.do_something_with_good_parsed_result(search_term, parsed_result)
            except Exception as e:
                self.logger.warn(msg="diagnose_parsed_result_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                return None
        except Exception as e:
            self.logger.warn(msg="parse_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
            return None

    def parsed_to_info_dict_process(self, search_term):
        try:
            try: # getting the parsed_dict for this search_term
                parsed_result = self.get_parsed_dict(search_term)
            except Exception as e:
                self.logger.warn(msg="get_parsed_dict_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                try: # getting the html for this search_term
                    html = self.get_html(search_term)
                except Exception as e:
                    self.logger.warn(msg="get_html_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                    return
                parsed_result = self.html_to_parsed_result(search_term, html)
                if parsed_result is None:
                    return None

            try: # getting info dict from parsed_result
                info_dict = self.get_info(parsed_result)
                try: # diagnose the info_dict to decide where to go from here
                    diagnosis_string = self.diagnose_info_dict(info_dict)
                    if diagnosis_string: # non-empty diagnosis_string means diagnosis failed
                        self.logger.warn(msg="Diagnosis failed for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                    else: # empty diagnosis_string means diagnosis succeeded
                        self.do_something_with_good_info_dict(search_term, info_dict)
                except Exception as e:
                    self.logger.warn(msg="diagnose_info_dict_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                    return
            except Exception as e:
                self.logger.warn(msg="get_info_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))
                return

        except Exception as e:
            self.logger.error(msg="untracked_exception for {} : {}.".format(to_utf8_or_bust(search_term), e.message))

    #### Class Utils
    def check_that_all_attributes_are_callable(self):
        pass
    #    for k,v in self.__dict__.items():
    #        assert hasattr(v,'__call__'), "%s is not callable" % k


    ##### Factories #################################################################################

    @classmethod
    def for_print_only(cls, html_folder):
        instance = ParseSearchTerms()
        instance._set_local_data_source(html_folder)
        instance._set_parse_and_get_info()
        instance._set_check_functions()
        instance._set_test_actions()
        instance.check_that_all_attributes_are_callable()
        return instance

    @classmethod
    def for_test_cloud(cls, log_filename):
        return cls.for_cloud(
            html_folder='ut-slurps/html/',
            parsed_results_folder='loc-data/dict/google_results_parse_result/',
            info_dict_folder='loc-data/dict/gresult_trinity_info/',
            log_filename = log_filename)

    @classmethod
    def for_semantics_cloud(cls, log_filename):
        return cls.for_cloud(
            html_folder='ut-slurps/html/',
            parsed_results_folder='ut-slurps/parsed',
            info_dict_folder='semantics-data/gresult_info_dict/',
            log_filename = KhanLogger.default_log_path_with_unique_name(log_filename)
        )

    @classmethod
    def for_local(cls,
                  html_folder=None,
                  parsed_results_folder=None,
                  info_dict_folder=None,
                  log_filename=None,
                  mother_root=None):
        instance = ParseSearchTerms()
        ds = util_data_source.for_global_local()
        html_folder = html_folder or ds.dir.khan_html
        parsed_results_folder = parsed_results_folder or ds.dir.khan_parsed_results
        info_dict_folder = info_dict_folder or ds.dir.khan_info_dict
        log_filename = log_filename or ds.f.khan_parse_search_terms_log
        instance._set_local_data_source(html_folder)
        instance._set_parse_and_get_info()
        instance._set_check_functions()
        instance._set_local_actions(
            parsed_results_folder=parsed_results_folder,
            info_dict_folder=info_dict_folder,
            log_filename=log_filename)
        instance.check_that_all_attributes_are_callable()
        return instance

    #@ensure_all_callable
    @classmethod
    def for_cloud(cls, html_folder, parsed_results_folder, info_dict_folder, log_filename):
        instance = cls()
        instance._set_cloud_data_source(html_folder=html_folder)
        instance._set_parse_and_get_info()
        instance._set_check_functions()
        instance._set_cloud_actions(
            parsed_results_folder=parsed_results_folder,
            info_dict_folder=info_dict_folder,
            log_filename=log_filename)
        instance.check_that_all_attributes_are_callable()
        return instance



    ##### Partial Factories #######################

    # ###################################################################

    def _set_local_data_source(self, html_folder):
        # get_and_save_html_if_not_saved_already
        html_data_accessor = DataAccessor(relative_root=html_folder,
                                          extension='', # '.html'
                                          force_extension=True,
                                          encoding='UTF-8',
                                          location=DataAccessor.LOCATION_LOCAL)
        self.get_html = html_data_accessor.loads

    def _set_cloud_data_source(self, html_folder=None, parsed_dict_folder=None):
        # getting htmls
        html_folder = html_folder or 'ut-slurps/html/'
        html_data_accessor = DataAccessor(relative_root=html_folder,
                                          extension='',
                                          force_extension=True,
                                          encoding='UTF-8',
                                          location=DataAccessor.LOCATION_S3)
        self.get_html = html_data_accessor.loads
        # # getting parsed_dicts
        # parsed_dict_folder = parsed_dict_folder or 'ut-slurps/parsed/'
        # parsed_dict_accessor = DataAccessor(relative_root=parsed_dict_folder,
        #                                   extension='.dict',
        #                                   force_extension=True,
        #                                   encoding='UTF-8',
        #                                   location=DataAccessor.LOCATION_S3)
        # self.get_parsed_dict = parsed_dict_accessor.loado

    def _set_parse_and_get_info(self):
        self.parse = parse_google.get_info_dict
        self.get_info = parse_gresult_info_retriever.GResultInfoRetriever.for_nres_words_domains().get_info

    def _set_check_functions(self):
        # # with checking for nres
        # self.diagnose_parsed_result = diagnose_nres_and_organic_results
        # self.diagnose_info_dict = diagnose_nres_words_domains
        # without checking for nres
        self.diagnose_parsed_result = diagnose_organic_results
        self.diagnose_info_dict = diagnose_words_domains

    def _set_local_actions(self, parsed_results_folder, info_dict_folder, log_filename):
        self.do_something_with_good_parsed_result = SaveDictLocally(save_folder=parsed_results_folder).save
        self.do_something_with_good_info_dict = SaveDictLocally(save_folder=info_dict_folder).save
        self.logger = KhanLogger(origin='parse search terms', level=logging.INFO)
        #self.log_failure = KhanLogger(origin=__name__).id_and_reason_log

    def _set_test_actions(self):
        self.do_something_with_good_parsed_result = print_parse_success
        self.do_something_with_good_info_dict = print_info_dict_success
        #self.logger = print_failure
        self.logger = PrintFailure()

    def _set_cloud_actions(self, parsed_results_folder, info_dict_folder, log_filename):
        self.do_something_with_good_parsed_result = SaveDictToS3(save_folder=parsed_results_folder).save
        self.do_something_with_good_info_dict = SaveDictToS3(save_folder=info_dict_folder).save
        #self.log_failure = KhanLogger(file_name=log_filename).id_and_reason_log
        self.logger = KhanLogger(origin='parse search terms')
        #self.log_failure = KhanLogger(__name__).id_and_reason_log # TODO: Not sure this is the way to log to s3

        ##### Matt's stuff
        # # Need to reverse order of args to comp   ly with signature of dswgpr call

        # self.do_something_with_good_parsed_result = S3.dumpo()#SaveDictLocally(save_folder=parsed_results_folder).save
        # self.do_something_with_good_info_dict = SaveDictLocally(save_folder=info_dict_folder).save
        # self.log_failure = KhanLogger(file_name=log_filename).id_and_reason_log


####### class ParseSearchTerms(object) ends here
#######################################################################################################################


### UTILS #############################################################################################################

##### checking functions

def diagnose_nres_and_organic_results(parse_result):
    suffix = ' (in diagnose_nres_and_organic_results)'
    if not isinstance(parse_result, dict):
        return 'parse_result not a dict' + suffix
    # MJM: put this back in!
    elif 'number_of_results' not in parse_result:
        return 'no number_of_results key' + suffix
    elif 'organic_results_list' not in parse_result:
        return 'no organic_results_list key' + suffix
    else:
        return ''

def diagnose_nres_words_domains(info_dict):
    suffix = ' (in diagnose_nres_words_domains)'
    if not isinstance(info_dict, dict):
        return 'info_dict not a dict'
    else:
        # MJM: put this back in!
        for k in ['number_of_results', 'term_stats', 'domain_names']:
        #for k in ['term_stats', 'domain_names']:
            if k not in info_dict:
                return 'no %s key'%k + suffix
            v = info_dict[k]
            if isinstance(v, pd.DataFrame):
                if len(v) == 0:
                    return '%s is an empty dataframe'%k + suffix
            elif not v:
                return '%s = None'%k + suffix
        return ''


def diagnose_organic_results(parse_result):
    suffix = ' (in diagnose_nres_and_organic_results)'
    if not isinstance(parse_result, dict):
        return 'parse_result not a dict' + suffix
    # MJM: put this back in!
    elif 'organic_results_list' not in parse_result:
        return 'no organic_results_list key' + suffix
    else:
        return ''

def diagnose_words_domains(info_dict):
    suffix = ' (in diagnose_nres_words_domains)'
    if not isinstance(info_dict, dict):
        return 'info_dict not a dict'
    else:
        # MJM: put this back in!
        for k in ['term_stats', 'domain_names']:
        #for k in ['term_stats', 'domain_names']:
            if k not in info_dict:
                return 'no %s key'%k + suffix
            v = info_dict[k]
            if isinstance(v, pd.DataFrame):
                if len(v) == 0:
                    return '%s is an empty dataframe'%k + suffix
            elif not v:
                return '%s = None'%k + suffix
        return ''


##### Success handlers

class SaveDictLocally(object):
    def __init__(self, save_folder):
        self.data_accessor = pfile_accessor.for_local(relative_root=save_folder,
                                          extension='.dict',
                                          force_extension=True)
    def save(self, name, parse_result):
        self.data_accessor.save(parse_result, name)

class SaveDictToS3(object):
    def __init__(self, save_folder):
        print("S3 pfile.Accessor created for %s" % save_folder)
        self.data_accessor = pfile_accessor.for_s3(relative_root=save_folder,
                                          extension='.dict',
                                          force_extension=True)
    def save(self, name, parse_result):
        self.data_accessor.save(parse_result, name)

def print_parse_success(search_term, parse_result):
    print("-----------------------------------------")
    print("%s" % search_term)
    print("  number of dict keys: %d" % len(parse_result))
    print("-----------------------------------------")
    print("")

def print_info_dict_success(search_term, info_dict):
    print("-----------------------------------------")
    print("%s" % search_term)
    print("  number_of_results: %d" % info_dict['number_of_results'])
    print("  # of unique terms: %d" % len(info_dict['term_stats']))
    print("  # of unique domains: %d" % len(info_dict['domain_names']))
    print("-----------------------------------------")
    print("")

##### Failure handlers
class PrintFailure(object):
    def print_it(self, *args, **kwargs):
        print("!!!!!!!!!!!!!!!!")
        for arg in args:
            print(arg)
        for k, v in kwargs:
            print("{}={}".format(k,v))

    def debug(self, *args, **kwargs):
        self.print_it(args, kwargs)
    def info(self, *args, **kwargs):
        self.print_it(args, kwargs)
    def warn(self, *args, **kwargs):
        self.print_it(args, kwargs)
    def error(self, *args, **kwargs):
        self.print_it(args, kwargs)

def print_failure(search_term, failure_type):
    print("!!!!!!!!!!!!!!!!")
    print("FAILURE for %s" % search_term)
    print("  --> %s" % failure_type)
    print("")


if __name__ == '__main__':
    # os.environ['MS_DATA']='/Users/mattjmorris/Dropbox/Dev/py/data/'
    # pst = ParseSearchTerms.for_semantics_cloud('sem_cloud_local')
    # s3 = S3('ut-slurps', 'to-slurp_raw')

    # import os
    print("begin")
    st = "5 star hotels leeds"
    pst = ParseSearchTerms.for_local(
        html_folder='s3/ut-slurps/html',
        parsed_results_folder='s3/ut-slurps/parsed',
        info_dict_folder='s3/semantics-data/gresult_info_dict',
        log_filename='log_parse_search_terms_02.txt')
    pst.process(st)
    print("done")
