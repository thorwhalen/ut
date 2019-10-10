
__author__ = 'thor'

print('''
    disp_html (but careful! save before use)
    BeautifulSoup
    bs3_BeautifulSoup
    ut.util.log.hms_message
    ut.webscrape.util.*
    filename_from_url
    url_from_filename
''')

from ut.util.importing import get_environment_variable
import logging

import urllib.request, urllib.error, urllib.parse
import urllib.parse
from bs4 import BeautifulSoup
from BeautifulSoup import BeautifulSoup as bs3_BeautifulSoup

import requests
from selenium import webdriver

# from selenium.webdriver.common.keys import Keys

import ut
import ut.parse.util
from ut.parse.util import disp_html
import ut.util.log
import ut.webscrape.util
from ut.webscrape.util import filename_from_url
from ut.webscrape.util import url_from_filename
from ut.parse.util import open_html_in_firefox

logging.basicConfig(filename=get_environment_variable('DEFAULT_LOG_FILE'), filemode='w', level=logging.DEBUG)

print("logging in use:\n   %s" % get_environment_variable('DEFAULT_LOG_FILE'))
