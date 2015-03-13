import requests

from ut.slurp.browser import Browser


class Google(Browser):

    URL = 'http://www.google.com/search'
    SEARCH_PARAMS = {
        'q': '',      # the search term
        'num': '10',  # the number of results per page this
        'pws': '0'    # personalization turned off
    }

    def __init__(self, proxies=None, auth=None, num_results=10):
        """
        Set up a browser-imitated slurper to slurp_raw Google search results.
        :type proxies: dict
        :type auth: requests.auth.HTTPProxyAuth
        :type num_results: into
        """
        Browser.__init__(self, proxies, auth)
        self.num_results = num_results

    @classmethod
    def with_anon(cls, num_results=10):
        return cls(cls.ANONYMIZER_PROXIES, cls.ANONYMIZER_AUTH, num_results)

    @classmethod
    def with_not_anon(cls, num_results=10):
        return cls(num_results=num_results)

    def safe_slurp(self, query):
        """
        This method is a replacement for the method 'slurp_raw'. The old method is deprecated and should be removed
        when it is verified that it is not used in a any notebooks.
        This method is 'safer' in that it raises an exception if the response is not OK.

        Slurp Google

        :param query: The search query
        :return: HTML result, or None if there was an HTTP error
        """
        search_params = self.SEARCH_PARAMS.copy()
        search_params.update({'q': query, 'num': self.num_results})
        headers = self.HEADERS.copy()
        headers['User-Agent'] = self.random_user_agent()

        r = requests.get(self.URL, headers=headers, params=search_params,
                         timeout=10.0, proxies=self.proxies, auth=self.auth)

        if not r.ok:
            raise ValueError('HTTP Error: {} for query {}'.format(r.status_code, query))
        else:
            return r.text

    def slurp(self, query):
        """
        !!! Deprecated !!!
        Use safe_slurp() instead

        Slurp Google

        :param query: The search query
        :return: HTML result, or None if there was an HTTP error
        """
        search_params = self.SEARCH_PARAMS.copy()
        search_params.update({'q': query, 'num': self.num_results})
        headers = self.HEADERS.copy()
        headers['User-Agent'] = self.random_user_agent()

        r = requests.get(self.URL, headers=headers, params=search_params,
                         timeout=10.0, proxies=self.proxies, auth=self.auth)

        if not r.ok:
            print('HTTP Error: {} for query {}'.format(r.status_code, query))
        else:
            return r.text