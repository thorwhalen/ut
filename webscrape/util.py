__author__ = 'thor'

from pattern.web import plaintext
# from ut.parse import html2text_formated
import os
import re
from bs4 import BeautifulSoup
import requests

params = dict()
params['html_ext_re'] = re.compile('.html$')


def html2text(text):
    return plaintext(text)
    # return html2text_formated.html2text(text).replace('**', '')


def filename_from_url(url, rm_http=True):
    if rm_http:
        url = url.replace('http://', '').replace('https://', '')
    return url.replace('/', '~').replace(':', '{') + '.html'


def url_from_filename(filename, prefix=''):
    return prefix + params['html_ext_re'].sub('', os.path.basename(filename).replace('~', '/').replace('{', ':'))


def what_ip_do_they_see(get_html_fun=lambda url: requests.get(url).text):
    b = BeautifulSoup(get_html_fun('http://www.whatismyip.com/'))
    return b.find('div', 'the-ip').text

# def filename_from_url_httpless(url)


# def url_to_filename(url):
