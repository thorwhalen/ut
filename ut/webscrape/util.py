"""Webscraping utils"""

__author__ = 'thor'

from contextlib import suppress
import os
import re

params = dict()
params['html_ext_re'] = re.compile('.html$')

ignore_module_error = suppress(ModuleNotFoundError, ImportError)


def html2text(text):
    with ignore_module_error:
        from pattern.web import plaintext as html2text

        return plaintext(text)
    with ignore_module_error:
        from html2text import html2text

        return html2text(text)


def filename_from_url(url, rm_http=True):
    if rm_http:
        url = url.replace('http://', '').replace('https://', '')
    return url.replace('/', '~').replace(':', '{') + '.html'


def url_from_filename(filename, prefix=''):
    return prefix + params['html_ext_re'].sub(
        '', os.path.basename(filename).replace('~', '/').replace('{', ':')
    )


with ignore_module_error:
    from bs4 import BeautifulSoup
    import requests

    def what_ip_do_they_see(get_html_fun=lambda url: requests.get(url).text):
        b = BeautifulSoup(get_html_fun('http://www.whatismyip.com/'))
        return b.find('div', 'the-ip').text

    def all_links_of_html(html):
        from bs4 import BeautifulSoup

        t = BeautifulSoup(html, features='lxml')
        return list(
            dict.fromkeys(  # this removes duplicates conserving order
                x.get('href')
                for x in t.find_all(
                    attrs={'href': re.compile('http.*')}, recursive=True
                )
            )
        )


# def filename_from_url_httpless(url)


# def url_to_filename(url):
