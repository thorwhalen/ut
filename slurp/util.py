__author__ = 'thor'

from ut.pdict.to import word_replacer

url_encode_yboss_map = {u' ': u'%20',
                 u'"': u'%22',
                 u'#': u'%23',
                 u'$': u'%24',
                 u'%': u'%25',
                 u'&': u'%26',
                 u'(': u'%28',
                 u')': u'%29',
                 u'*': u'%2A',
                 u'+': u'%2B',
                 u',': u'%2C',
                 u'/': u'%2F',
                 u':': u'%3A',
                 u';': u'%3B',
                 u'<': u'%3C',
                 u'=': u'%3D',
                 u'>': u'%3E',
                 u'?': u'%3F',
                 u'@': u'%40',
                 u'[': u'%5B',
                 u'\\': u'%5C',
                 u']': u'%5D',
                 u'^': u'%5E',
                 u'`': u'%60',
                 u'{': u'%7B',
                 u'|': u'%7C',
                 u'}': u'%7D'}

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
