__author__ = 'thor'

from bs4 import BeautifulSoup
from functools import wraps


@wraps(
    BeautifulSoup.find_all,
    assigned=('__module__', '__qualname__', '__annotations__', '__name__'),
)
def gen_find(tag, *args, **kwargs):
    """Does what BeautifulSoup.find_all does, but as an iterator.
        See find_all documentation for more information."""
    if isinstance(tag, str):
        tag = BeautifulSoup(tag)
    next_tag = tag.find(*args, **kwargs)
    while next_tag is not None:
        yield next_tag
        next_tag = next_tag.find_next(*args, **kwargs)


from ut.pdict.to import word_replacer

url_encode_yboss_map = {
    ' ': '%20',
    '"': '%22',
    '#': '%23',
    '$': '%24',
    '%': '%25',
    '&': '%26',
    '(': '%28',
    ')': '%29',
    '*': '%2A',
    '+': '%2B',
    ',': '%2C',
    '/': '%2F',
    ':': '%3A',
    ';': '%3B',
    '<': '%3C',
    '=': '%3D',
    '>': '%3E',
    '?': '%3F',
    '@': '%40',
    '[': '%5B',
    '\\': '%5C',
    ']': '%5D',
    '^': '%5E',
    '`': '%60',
    '{': '%7B',
    '|': '%7C',
    '}': '%7D',
}

url_encode_yboss = word_replacer(url_encode_yboss_map, inter_token_re='')

# def mk_url_encode_yboss_map():
#     import requests
#     import re
#     from bs4 import BeautifulSoup
#     r = requests.get('http://developer.yahoo.com/boss/search/boss_api_guide/reserve_chars_esc_val.html')
#     b = BeautifulSoup(r.text)
#     t = b.find('div', attrs={'class':'informaltable'}).find('tbody')
#     tt = t.findAll('tr')[1:]
#
#     w = [x.findAll('td') for x in tt]
#     mapfrom = [x[0] for x in w]
#     mapto = [x[1] for x in w]
#
#     mapfrom = [x.text for x in mapfrom]
#     mapfrom = [re.findall('(?<=\().(?=\))',x)[0] for x in mapfrom]
#     mapto = [x.text for x in mapto]
#     mapto = [x.replace('\n','') for x in mapto]
#
#     map_rtf_3986 = {f:t for f,t in zip(mapfrom,mapto)}
#     return map_rtf_3986
