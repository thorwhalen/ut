__author__ = 'thorwhalen'

from unidecode import unidecode
import os
import re
from bs4 import BeautifulSoup
from StringIO import StringIO
#import ut.pfile

import subprocess
import tempfile

from HTMLParser import HTMLParser
from bs4.element import Tag
import ut.util.ulist as util_ulist
import ut.pstr.trans as pstr_trans

import ut.pfile.to
import ut.pstr.to
import ut.pstr.trans

from IPython.display import HTML as disp_html_fun
from IPython.core.display import display as ipython_display



RE_HAS_NEW_LINE = re.compile('\n|\r')
RE_NUM_SEP = "[,\.\s]"
RE_NUMBER = "\d[\d,\.\s]+\d|\d"
CRE_NUMBER = re.compile(RE_NUMBER)



# def tidy_html(html):
#     pass

def clean_html_01(html):
    from lxml import etree
    tree   = etree.HTML(html.replace('\r', ''))
    return '\n'.join([ etree.tostring(stree, pretty_print=True, method="xml")
                              for stree in tree ])


def disp_html(html):
    if not isinstance(html, basestring):
        try:
            html = html.renderContents()
        except AttributeError or TypeError:
            pass
    try:
        html_disp = disp_html_fun(html)
    except BaseException as e:
        html_disp = "failed<p>" + e.message
    ipython_display(html_disp)


def strip_spaces(s):
    return pstr_trans.strip(s.replace(u'\xa0', ' '))


def list_to_exp(str_list, term_padding_exp=r'\b', compile=True):
    """
    Returns a regular expression (compiled or not) that will catch any of the strings of the str_list.
    Each string of the str_list will be surrounded by term_padding_exp (default r'\b' forces full word matches).
    Note: Also orders the strings according to length so that no substring will overshadow a superstring.
    """
    str_list = util_ulist.sort_as(str_list, map(len, str_list), reverse=True)
    exp = term_padding_exp + '(' + '|'.join(str_list) + ')' + term_padding_exp
    if compile:
        return re.compile(exp)
    else:
        return exp


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def printable_text(tag, sep="\n"):
    """
    sometimes, when trying to print a string, I get the error:
        'ascii' codec can't encode character X in position Y: ordinal not in range(128)
    this is a hack to get rid of this by using unidecode
    """
    try:
        return "{}".format(sep.join(tag.strings))
    except:
        return "{}".format(unidecode(sep.join(tag.strings)))

def print_text(tag,sep="\n"):
    try:
        print "{}".format(sep.join(tag.strings))
    except:
        print "{}".format(unidecode(sep.join(tag.strings)))

def print_names_and_attrs(tag,max_depth=1,indent=""):
    """
    prints the name and attrs of a node and its descendance (up to the specified max_depth)
    """
    if isinstance(tag,list):
        for t in tag:
            print_names_and_attrs(t,max_depth=max_depth,indent=indent)
    else:
        print indent + tag.name + " " + str(tag.attrs)
        if max_depth > 0:
            print_names_and_attrs(list(tag.children),max_depth-1,indent+"  ")

def open_in_firefox(url):
    if isinstance(url, str):
        # -new-tab
        print "opening {}".format(url)
        os.system('open -a FireFox "{}"'.format(url))
    elif isinstance(url, list):
        for u in url:
            open_in_firefox(url)


def open_html_in_firefox(html):
    if isinstance(html, Tag) or isinstance(html, BeautifulSoup):
        html = html.renderContents()
    try:
        ut.pstr.to.file(ut.pstr.trans.str_to_utf8_or_bust(html), 'tmp_open_html_in_firefox.html')
    except:
        ut.pstr.to.file(html, 'tmp_open_html_in_firefox.html')
    open_in_firefox('tmp_open_html_in_firefox.html')


def pretty(soup):
    from lxml import etree, html
    print(etree.tostring(html.fromstring(str(soup)), encoding='unicode', pretty_print=True))

########### translators

def x_to_soup(input):
    this_input_type = input_type(input)
    if this_input_type:
        if this_input_type=='soup':
            return input
        elif this_input_type=='file':
            input = ut.pfile.to.string(input)
            # elif this_input_type=='url':
        #     input = urlopen(this_input_type).read()
        # else assert the input is the html string itself
        assert isinstance(input,basestring),'input must be a soup, file, url, or html string'
        # MJM - changing parser to html.parser from default of the faster lxml lib, for now, since lxml seems to be
        # failing, at least with python 2.7.5
        res = BeautifulSoup(input, "html.parser")
        return res
    else:
        return None

def x_to_file(input):
    this_input_type = input_type(input)
    if this_input_type:
        if this_input_type=='file' or this_input_type=='StringIO':
            return input
        elif this_input_type=='string':
            input = StringIO(input)
            # elif this_input_type=='url':
            #     input = StringIO(urlopen(input).read())
            # else assert the input is the html string itself
        return input
    else:
        return None

def input_type(input):
    """
    returns the type of input (file, string, url, soup (could be Tag) or None)
    """
    if isinstance(input,basestring):
        if os.path.exists(input):
            return 'file'
        elif RE_HAS_NEW_LINE.search(input):
            return 'string'
        else:
            return 'url'
    elif isinstance(input,StringIO):
        return 'StringIO'
    elif isinstance(input,BeautifulSoup) or isinstance(input,Tag):
        return 'soup'
    else:
        return None

def extract_section(soup, attrs={}, name='div', all=False):
    """
    gets the bs4.element.Tag (or list thereof) of a section specified by the attrs (dict or list of dicts)
    and extracts (rips out of the tree (soup)) the found tags as well (to save memory)
    """
    if soup: # soup is non-None
        if all==False:
            if isinstance(attrs,dict):
                t = soup.find(name=name, attrs=attrs)
                if t: t.extract()
                return t
            else:
                t = soup
                for ss in attrs:
                    t = t.find(name=name, attrs=ss)
                if t: t.extract()
                return t
        else:
            if isinstance(attrs,dict):
                t =  soup.findAll(name=name, attrs=attrs)
                if t: t.extract()
                return t
            else: # not sure how to handle this, so I'm forcing exit
                print "haven't coded this yet"
                return None
    else:
        return None