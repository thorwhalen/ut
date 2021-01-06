"""yboss slurping"""
__author__ = 'thor'

import os
from ut.util.importing import get_environment_variable
import oauth2
import time
import httplib2
import json
import pandas as pd

import ut as ms
from ut.slurp.util import url_encode_yboss
from ut.pdict.get import mk_fixed_coordinates_value_getter
from ut.webscrape.yboss import YbossText



yboss_root_url = "http://yboss.yahooapis.com"
yboss_search_root_url = yboss_root_url + '/ysearch/'

default_universal_args = {
    'start': 0,
    'count': 50,
    'market': 'en-us',
    'format': 'json'
}

service_default_req_args = {
    'web': {'count': 50, 'abstract': 'long', 'style': 'raw'},  # filter, type, view, title, url, sites
    'limitedweb': {'count': 50, 'abstract': 'long', 'style': 'raw'},  # filter, type, view, title, url, sites
    'news': {'count': 50, 'style': 'raw'},  # age, sort, title, url
    'blogs': {'count': 20, 'style': 'raw'},  # age, sort, count, title, url
    'related': {'count': 10},  # age, sort, count, title, url
    'images': {'count': 35}  # filter, queryfilter, dimensions, referurl, url
}

default_yboss_attrs = {
    'oauth_consumer_key': get_environment_variable('MON_YB_KEY'),
    'oauth_consumer_secret': get_environment_variable('MON_YB_SECRET'),
    'default_service': 'limitedweb',
    'default_params': {},
    'default_save_folder': os.getcwd()
}

service_list = ['limitedweb', 'web', 'blogs', 'news', 'related', 'images']

major_cols = ['query', 'position', 'title', 'abstract', 'dispurl', 'num_of_slurped_results', 'author']
minor_cols = ['date', 'url', 'clickurl']


class Yboss(object):

    default_yboss_attrs = {
        'oauth_consumer_key': get_environment_variable('MON_YB_KEY'),
        'oauth_consumer_secret': get_environment_variable('MON_YB_SECRET'),
        'default_service': 'limitedweb',
        'default_params': {},
        'default_save_folder': os.getcwd()
    }

    def __init__(self, **kwargs):
        self.__dict__.update(Yboss.default_yboss_attrs)
        self.__dict__.update(kwargs)
        self.consumer = oauth2.Consumer(key=self.oauth_consumer_key, secret=self.oauth_consumer_secret)

    ####################################################################################
    ###### SLURPERS ####################################################################
    def slurp_raw(self, query, service=None, params=None):
        service, params = self.fill_with_defaults(service, params)
        url = self.url(query, service, params)

        request_params = {
            'oauth_version': '1.0',
            'oauth_nonce': oauth2.generate_nonce(),
            'oauth_timestamp': int(time.time()),
        }
        oauth_request = oauth2.Request(method='GET', url=url, parameters=request_params)
        oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), self.consumer, None)
        oauth_header = oauth_request.to_header(realm='yahooapis.com')

        # Get search results
        http = httplib2.Http()
        resp, content = http.request(url, 'GET', headers=oauth_header)
        return {'content': content, 'resp': resp}
        # return "{'resp': %s, 'content': %s}" % (resp, content)

    def slurp_content(self, query, service=None, params=None):
        resp_content = self.slurp_raw(query, service=service, params=params)
        return resp_content['content']

    def slurp_content_as_dict(self, query, service=None, params=None):
        return json.loads(self.slurp_content(query, service=service, params=params))

    def slurp_content_and_save(self, query, service=None, params=None, filepath=None):
        filepath = self.get_filepath_for_params(query=query, service=service, params=params, filepath=filepath)
        resp_content = self.slurp_raw(query, service=service, params=params)
        json.dump(resp_content['content'], open(filepath, 'w'))

    def slurp_df_and_save(self, query, service=None, params=None, filepath=None, n_pages=1):
        filepath = self.get_filepath_for_params(query=query, service=service, params=params, filepath=filepath)
        df = self.slurp_results_df_multiple_pages(query=query, service=service, params=params, n_pages=n_pages)
        pd.to_pickle(df, filepath)
        return df

    def get_df(self, query, service=None, params=None, filepath=None, n_pages=1, overwrite=False):
        filepath = self.get_filepath_for_params(query=query, service=service, params=params, filepath=filepath)
        if not overwrite and os.path.exists(filepath):
            return pd.read_pickle(filepath)
        else:
            return self.slurp_df_and_save(query=query, service=service, params=params, n_pages=n_pages)

    def slurp_results_df(self, query, service=None, params=None):
        content_dict = json.loads(self.slurp_content(query, service=service, params=params))
        content_dict = self.get_item(content_dict)
        return self.content_to_results_df(content_dict)

    def content_to_results_df(self, content_dict):
        df = pd.DataFrame(content_dict['results'])
        start_position = int(content_dict['start'])
        df['position'] = list(range(start_position, start_position+len(df)))
        df._metadata = {'totalresults': int(content_dict['totalresults'])}
        return df

    def slurp_results_df_multiple_pages(self, query, service=None, params=None, n_pages=5):
        service, params = self.fill_with_defaults(service, params)
        df = pd.DataFrame()
        new_df = pd.DataFrame()
        for i in range(n_pages):
            # print "slurping %d/%d" % (i, n_pages-1)
            try:
                new_df = self.slurp_results_df(query, service=service, params=params)
                df = pd.concat([df, new_df])
            except:
                break
            params['start'] += params['count']
        df._metadata = new_df._metadata
        return df


    ####################################################################################
    ###### UTILS #######################################################################


    ####################################################################################
    ###### CONTENT ACCESSORS ###########################################################

    def load_json_dict(self, filepath):
        filepath = self.get_filepath(filepath)
        return json.loads(json.load(open(filepath, 'r')))

    def get_service_results_getter(self, service):
        service = service or self.default_save_folder
        return mk_fixed_coordinates_value_getter(['bossresponse', service])

    @classmethod
    def get_results(cls, content_dict):
        return Yboss.get_item(content_dict)['results']

    # def get_results

    @classmethod
    def get_totalresults(cls, content_dict):
        return int(Yboss.get_item(content_dict)['totalresults'])

    @classmethod
    def get_item(cls, content_dict):
        content_dict = content_dict['bossresponse']
        return content_dict[list(content_dict.keys())[0]]

    @classmethod
    def mk_fixed_coordinates_value_getter(cls, coord_list):
        return mk_fixed_coordinates_value_getter(['bossresponse'] + coord_list)

    ####################################################################################
    ###### UTILS #######################################################################
    def url(self, query, service=None, params=None):
        service = service or self.default_service
        params = params or self.default_params
        return yboss_search_root_url + self.rel_url(query=query, service=service, params=params)

    def rel_url(self, query, service=None, params=None):
        service = service or self.default_service
        params = params or self.default_params
        params = Yboss.mk_req_params(service, params)
        return "%s?q=%s%s" % (service, self.url_encode_str(query), self.url_encode_params(params))

    def get_filename_for_query(self, query, service=None, params=None):
        return self.rel_url(query, service=service, params=params).replace('?', '--')

    def get_filepath(self, filespec):
        if os.path.exists(filespec):
            file_path = filespec
        else:
            file_path = os.path.join(self.default_save_folder, filespec)
            if not os.path.exists(file_path):
                # assume it's a query, and derive what the filepath should be
                file_path = os.path.join(self.default_save_folder, self.get_filename_for_query(filespec))
        return file_path

    def get_filepath_for_params(self, query, service=None, params=None, filepath=None):
        filepath = filepath or self.default_save_folder
        if os.path.isdir(filepath):  # if filepath is a directory, need to make a filename for it
            filepath = os.path.join(filepath, self.get_filename_for_query(query, service=service, params=params))
        return filepath

    def fill_with_defaults(self, service=None, params=None):
        service = service or self.default_service
        params = Yboss.mk_req_params(service, params)
        return service, params

    @classmethod
    def mk_req_params(cls, service, params=None):
        params = params or {}
        return dict(
            dict(default_universal_args, **service_default_req_args[service]),
            **params)

    @classmethod
    def url_encode_str(cls, s):
        return url_encode_yboss(s)

    @classmethod
    def url_encode_params(cls, params):
        u = ''
        for p, v in params.items():
            if isinstance(v, str):
                u += '&%s=%s' % (p, v)
            else:
                u += '&%s=%s' % (p, str(v))
        return u

    @classmethod
    def print_some_resources(cls):
        print('''
            guide to yahoo BOSS: http://developer.yahoo.com/boss/search/boss_api_guide/index.html
            pricing (by service): http://developer.yahoo.com/boss/search/#pricing
            services: web, limitedweb, images, news, blogs, related
            response fields: http://developer.yahoo.com/boss/search/boss_api_guide/webv2_response.html
            market and languages: http://developer.yahoo.com/boss/search/boss_api_guide/supp_regions_lang.html
                ''')

    @classmethod
    def process_df(cls, df):
        df['dispurl'] = df['dispurl'].map(YbossText.remove_html_bold)
        df['title'] = df['title'].apply(YbossText.html2text)
        df['abstract'] = df['abstract'].apply(YbossText.html2text)
        df = ms.daf.manip.reorder_columns_as(df, major_cols + minor_cols)
        df = df.reset_index(drop=True)
        return df


