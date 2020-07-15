import pandas as pd


def url_to_html_func(kind='requests'):
    """Get a url_to_html function of a given kind."""
    url_to_html = None
    if kind == 'requests':
        import requests

        def url_to_html(url):
            r = requests.get(url)
            if r.status_code != 200:
                print(f"An error occured. Returning the response object for you to analyze: {r}")
                return r
            return r.content
    elif kind == 'chrome':
        from selenium import webdriver
        from time import sleep

        def url_to_html(url, wait=2):
            b = webdriver.Chrome()
            b.get(url)
            if isinstance(wait, (int, float)):
                sleep(wait)
            html = b.page_source
            b.close()
            return html
    else:
        raise ValueError(f"Unknown url_to_html value: {url_to_html}")
    assert callable(url_to_html), "Couldn't make a url_to_html function"

    return url_to_html


def get_tables_from_url(url, url_to_html='requests'):
    """Get's a list of pandas dataframes from tables scraped from a url.
    Note that this will only work with static pages. If the html needs to be rendered dynamically,
    you'll have to get your needed html otherwise (like with selenium).

    >>> url = 'https://en.wikipedia.org/wiki/List_of_musical_instruments'
    >>> tables = get_tables_from_url(url)

    If you install selenium and download a chromedriver,
    you can even use your browser to render dynamic html.
    Say, to get updated coronavirus stats without a need to figure out the API
    (I mean, why have to figure out the language of an API, when someone already did that
    for you in their webpage!!):

    ```python
    url = 'https://www.worldometers.info/coronavirus/?utm_campaign=homeAdvegas1?'
    tables = get_tables_from_url(url, url_to_html='chrome')
    ```

    To make selenium work:
    ```
        pip install selenium
        Download seleniumdriver here: https://chromedriver.chromium.org/
        Uzip and put in a place that's on you PATH (run command `echo $PATH` for a list of those places)
    ```
    """
    if not callable(url_to_html):
        url_to_html = url_to_html_func(url_to_html)
    return pd.read_html(url_to_html(url))
