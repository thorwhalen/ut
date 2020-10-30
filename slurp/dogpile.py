import requests

from .browser import Browser


class Dogpile(Browser):

    URL = 'http://www.dogpile.com/search/web'
    SEARCH_PARAMS = {
        'q': '',      # the search term
    }

    def __init__(self, proxies=None, auth=None):
        """
        Set up a browser-imitated slurper to slurp_raw Dogpile search results.
        :type proxies: dict
        :type auth: requests.auth.HTTPProxyAuth
        """
        Browser.__init__(self, proxies, auth)

    def slurp(self, query):
        """
        Slurp Google

        :param query: The search query
        :return: HTML result, or None if there was an HTTP error
        """
        search_params = self.SEARCH_PARAMS.copy()
        search_params.update({'q': query})
        headers = self.HEADERS.copy()
        headers['User-Agent'] = self.random_user_agent()

        r = requests.get(self.URL, headers=headers, params=search_params, timeout=10.0, proxies=self.proxies, auth=self.auth)

        if not r.ok:
            print(('HTTP Error: {} for query {}'.format(r.status_code, query)))
        else:
            return r.text