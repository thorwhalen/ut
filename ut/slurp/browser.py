import json
import os
from ut.util.importing import get_environment_variable
import random
from requests.auth import HTTPProxyAuth
import urllib.request, urllib.error, urllib.parse
import requests
import re

from selenium import webdriver
import selenium.webdriver.common.keys

import ut.util.pobj


class Browser(object):
    """ Base class for slurpers """

    # Load up scraper config stuff from config file upon loading class definition
    CONFIG = json.load(open(os.path.dirname(__file__) + '/config.json'))
    USER_AGENTS = CONFIG['USER_AGENTS']
    HEADERS = CONFIG['HEADERS']
    try:
        ANONYMIZER_AUTH = HTTPProxyAuth(
            get_environment_variable('VEN_ANONYMIZER_LOGIN'),
            get_environment_variable('VEN_ANONYMIZER_PASS'),
        )
        ANONYMIZER_PROXIES = {
            'http': get_environment_variable('VEN_ANONYMIZER_PROX_HTTP')
        }  # ,'https': get_environment_variable('VEN_ANONYMIZER_PROX_HTTPS']}
    except KeyError:
        print(
            'VEN_ANONYMIZER_LOGIN, VEN_ANONYMIZER_PASS, and/or VEN_ANONYMIZER_PROX_HTTP missing from environment'
        )
    try:
        PROXYMESH_AUTH = requests.auth.HTTPProxyAuth(
            get_environment_variable('PROXYMESH_USER'),
            get_environment_variable('PROXYMESH_PASS'),
        )
        PROXYMESH_PROXIES = {'http': 'http://us.proxymesh.com:31280'}
    except KeyError:
        print('PROXYMESH_USER and/or PROXYMESH_PASS missing from environment')

    def __init__(self, **kwargs):
        """
        Creates an instance that will use proxies and authorization if proxies & auth are provided.
        :type proxies: dict
        :type auth: requests.auth.HTTPProxyAuth
        """
        # CONFIG = json.load(open(os.path.dirname(__file__) + "/config.json"))
        default_kwargs = {
            'get_header_fun': 'header_with_random_firefox_user_agent',
            'random_agents': [],
            'header': {
                'User-Agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'close',
                'DNT': '1',
            },
            'proxies': None,
            'auth': None,
            'timeout': 10.0,
        }
        self = ut.util.pobj.set_attributes(self, kwargs, default_kwargs)
        # self.random_agents = []
        # self.proxies = proxies
        # self.auth = auth
        # self.timeout = 10.0
        # self.webdriver = None

    @classmethod
    def options(cls):
        print(
            '''
            get_header_fun:
                'header_with_random_firefox_user_agent' (default)
                'fixed_header'
        '''
        )

    def get_header(self):
        if self.get_header_fun == 'header_with_random_firefox_user_agent':
            return self.header_with_random_user_agent(
                filter_in_user_agents='^Mozilla.*'
            )
        elif self.get_header_fun == 'fixed_header':
            return self.header
        else:
            raise ValueError('Unknown get_header_fun value')

    def header_with_random_user_agent(self, filter_in_user_agents=None):
        """
        Returns a header with a random user agent
        """
        headers = self.HEADERS.copy()
        headers['User-Agent'] = self.random_user_agent(
            filter_in_user_agents=filter_in_user_agents
        )
        return headers

    def random_user_agent(self, filter_in_user_agents=None):
        """
        Returns a random user agent from the full list of user agents.
        Cycles through all agents before re-sampling from the full list again.
        """
        if not self.random_agents:
            self.random_agents = self.USER_AGENTS[:]
            if filter_in_user_agents:
                self.random_agents = list(
                    filter(re.compile(filter_in_user_agents).search, self.random_agents)
                )
        random.shuffle(self.random_agents)
        return self.random_agents.pop()

    def get_html_through_tor_unfinished(self, url):
        UserWarning("This hasn't really be coded yet...")
        proxy_support = urllib.request.ProxyHandler({'http': '127.0.0.1:8118'})
        opener = urllib.request.build_opener(proxy_support)
        opener.addheaders = [('User-agent', self.random_user_agent())]
        return opener.open(url).read()

    def get_html_through_requests(self, url, url_params={}, timeout=None):
        r = self.get_response_through_requests(
            url=url, url_params=url_params, timeout=timeout
        )
        # return the text if no error
        if not r.ok:
            raise ValueError(
                'HTTP Error: {} for url {} (headers="{}"'.format(
                    r.status_code, url, str(r.headers)
                )
            )
        else:
            return r.text

    def get_response_through_requests(self, url, url_params={}, timeout=None):
        timeout = timeout or self.timeout
        header = self.header_with_random_user_agent()
        # get the content for the url
        r = requests.get(
            url,
            headers=header,
            params=url_params,
            timeout=timeout,
            proxies=self.proxies,
            auth=self.auth,
        )
        return r

    def get_html_through_selenium(self, url, url_params={}, timeout=None):
        ValueError("You haven't written this yet!!")
        pass
        # timeout = timeout or self.timeout
        # header = self.header_with_random_user_agent()
        # # get the content for the url
        # r = requests.get(url, headers=header, params=url_params,
        #                  timeout=timeout, proxies=self.proxies, auth=self.auth)
        # # return the text if no error
        # if not r.ok:
        #     raise ValueError('HTTP Error: {} for url {} (user-agent="{}"'.format(r.status_code, url, header['User-Agent'] ))
        # else:
        #     return r.text

    @classmethod
    def with_ven_anonymizer(cls):
        return Browser(
            proxies={'http': get_environment_variable('VEN_ANONYMIZER_PROX_HTTP')},
            auth=HTTPProxyAuth(
                get_environment_variable('VEN_ANONYMIZER_LOGIN'),
                get_environment_variable('VEN_ANONYMIZER_PASS'),
            ),
        )

    @classmethod
    def with_proxymesh(cls):
        return Browser(proxies=cls.PROXYMESH_PROXIES, auth=cls.PROXYMESH_AUTH)

    @classmethod
    def firefox_selenium(cls, **kwargs):
        default_kwargs = {'webdriver': webdriver.Firefox()}
        kwargs = dict(default_kwargs, **kwargs)
        return Browser(**kwargs)


class Selenium(Browser):
    def __init__(self, **kwargs):
        default_kwargs = {
            'header': {
                'User-Agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'close',
                'DNT': '1',
            },
            'random_agents': [],
            'proxies': None,
            'auth': None,
            'timeout': 10.0,
            'webdriver': 'firefox',
        }
        self = ut.util.pobj.set_attributes(
            self, dict(default_kwargs, **kwargs), Browser().__dict__
        )
        if self.webdriver == 'firefox':
            self.webdriver = webdriver.Firefox()
        elif self.webdriver == 'phantomjs':
            self.webdriver = webdriver.PhantomJS()

    def goto_url(self, url):
        self.webdriver.get(url)

    def get_html(self, url=None):
        if url:
            self.goto_url(url)
        return self.webdriver.page_source
