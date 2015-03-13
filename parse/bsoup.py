__author__ = 'thorwhalen'
"""
functions that work on soup, soup tags, etc.
"""

from ut.pgenerator.get import last_element
from tempfile import mkdtemp
import os
import ut.pstr.to as strto
import ut.parse.util as parse_util
import ut.pstr.trans as pstr_trans

def root_parent(s):
    return last_element(s.parents)

def open_tag_in_firefox(tag):
    save_file = os.path.join(mkdtemp(),'tmp.html')
    strto.file(tag.prettify(), save_file)
    parse_util.open_in_firefox(save_file)

def add_text_to_parse_dict(soup, parse_dict, key, name, attrs, text_transform=pstr_trans.strip):
    tag = soup.find(name=name, attrs=attrs)
    if tag:
        if text_transform:
            parse_dict[key] = text_transform(tag.text)
        else:
            parse_dict[key] = tag.text
    return parse_dict