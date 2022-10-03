__author__ = 'thor'

import ut as ms
import re
import requests
from bs4 import BeautifulSoup
import os
import ut.pfile.to


def get_multiple_template_dicts(source):
    templates = dict()
    if isinstance(source, str):
        if (
            not re.compile('\n|\t').match(source) and len(source) < 150
        ):  # assume it's a filepath or url...
            if os.path.exists(source):
                source = ms.pfile.to.string(source)
            else:
                source = requests.get(source).text  # ... and get the html
        soup = BeautifulSoup(source)
    table_soup_list = soup.find_all('table')
    print('Found %d tables...' % len(table_soup_list))
    for table_soup in table_soup_list:
        try:
            tt = mk_simple_template_dict(table_soup)
            templates[tt['table']['id']] = tt
        except Exception:
            raise
    print('... could extract a template from %d of these' % len(templates))
    return templates


def mk_simple_template_dict(table_soup):
    """
    Tries to create a template dict from html containing a table (should feed it with soup.find('table') for example)
    This function assumes that all thead cells are formated the same, and all tbody rows are formated the same
    """
    # global table attributes
    bb = table_soup
    glob = dict()
    glob['id'] = bb.attrs.get('id')
    glob['summary'] = ''
    glob['style'] = parse_style(bb.attrs.get('style'))
    glob
    # thead attributes
    bb = table_soup.find('thead').find('th')
    thead = dict()
    thead['scope'] = bb.attrs.get('scope')
    thead['style'] = parse_style(bb.attrs.get('style'))
    thead
    # tbody attributes
    bb = table_soup.find('tbody').find('tr').find('td')

    tbody = dict()
    tbody['style'] = parse_style(bb.attrs.get('style'))
    tbody

    return {'table': glob, 'thead': thead, 'tbody': tbody}


def parse_style(style_string):
    if style_string:
        style_dict = dict()
        t = re.compile('[^:]+:[^;]+;').findall(
            style_string.replace('\n', '').replace('\t', '')
        )
        t = [x.replace(';', '') for x in t]
        t = [x.split(':') for x in t]
        for i in range(len(t)):
            for ii in range(len(t[i])):
                t[i][ii] = t[i][ii].strip()
        style_dict = dict()
        for ti in t:
            style_dict[ti[0]] = ti[1]
        return style_dict
    else:
        return None
