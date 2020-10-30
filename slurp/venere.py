__author__ = 'thorwhalen'


import urllib.request, urllib.error, urllib.parse
from collections import OrderedDict
import pandas as pd

VENERE_URL = "www.venere.com"
HTTP_PREFIX = "http://"
HTTP_VENERE_ROOT_URL = "http://www.venere.com/"

from bs4 import BeautifulSoup

def get_destination_page_info(geoid):
    response = urllib.request.urlopen(geoid_to_destination_url(geoid))
    finalurl = response.geturl()
    finalurl = finalurl.replace(VENERE_URL,'')
    finalurl = finalurl.replace(HTTP_PREFIX,'')
    html = response.read()
    s = BeautifulSoup(html)
    t = s.find(name='div', attrs={'id':'matchinghotels'})
    tt = t.find(name='p', attrs={'id':'results'})
    num_of_hotels = int(tt.getText())
    response.close()
    return {'geo_id':geoid, 'num_of_hotels':num_of_hotels, 'url_stub':finalurl}

def geoid_to_destination_url(geoid):
    return HTTP_VENERE_ROOT_URL + 'search/?geoid=%d' % geoid
