__author__ = 'thorwhalen'


from bs4 import BeautifulSoup
import re
import os.path
from urllib2 import urlopen
from pfile import to
from lxml import etree
import tldextract
import parse.util as util
from urlparse import urlparse, parse_qs
from analyzer import extract_section

# RE_HAS_NEW_LINE = re.compile('\n|\r')
RE_NUM_SEP = "[,\.\s]"
RE_NUMBER = "\d[\d,\.\s]+\d|\d"
CRE_NUMBER = re.compile(RE_NUMBER)
XP_NRESULT = '//*[@id="ab_ps_r"]/text()'
# PARSE_DEF = {
#     'center_col':{
#         'expand':{
#             'taw':{'attrs':{'id':'taw'},'name':'span'},
#             'res':{'attrs':{'id':'res'},'name':'div'},
#             'extrares':{'attrs':{'id':'extrares'},'name':'div'}
#         }}}

##################################################################################
# IMPORTANT FUNCTIONS
##################################################################################


def mk_gresult_tag_dict(input):
    """
    mk_result_dict(input)
    takes a soup, html string, or filename of google result htmls as an input
    and returns a dict containing components of the html we're interested in
    """
    input = util.x_to_soup(input)
    d = dict()

    # number of results
    resultStats = input.find(name='div',attrs={'id':'resultStats'})
    if resultStats: d['_resultStats'] = resultStats

    # center_col
    center_col = input.find(name='div',attrs={'id':'center_col'})
    if center_col:
        d['_center_col'] = center_col
        # tads
        tads = d['_center_col'].find(name='div',attrs={'id':'tads'})
        if tads:
            d['_tads'] = tads
            # top_ads
            top_ads = d['_tads'].findAll('li')
            if top_ads:
                d['_top_ads_list'] = top_ads
        # res
        res = d['_center_col'].find(name='div',attrs={'id':'res'})
        if res:
            d['_res'] = res
            # searchInstead
            topstuff = d['_res'].find(name='div',attrs={'id':'topstuff'})
            if topstuff:
                d['_topstuff'] = topstuff
                # spell
                spell = d['_topstuff'].find(name='a',attrs={'class':'spell'})
                if spell: d['_spell'] = spell
            # search
            search = d['_res'].find(name='div',attrs={'id':'search'})
            if search:
                d['_search'] = search
                # ires
                ires = d['_search'].find(name='div',attrs={'id':'ires'})
                if ires:
                    d['_ires'] = ires
                    # organicResults
                    organic_results = d['_ires'].findAll('li')
                    if organic_results:
                        d['_organic_results_list'] = organic_results

        # related_search
        after_res = d['_res'].nextSibling
        if after_res:
            related_search = after_res.find('table')
            if related_search:
                d['_related_search'] = related_search

    # rhs_block
    rhs_block = input.find(name='div',attrs={'id':'rhs_block'})
    if rhs_block:
        d['_rhs_block'] = rhs_block
        rhs_ads = [] # initializing
        # rhs_ads from mbEnd
        mbEnd = d['_rhs_block'].find(name='div',attrs={'id':'mbEnd'})
        if mbEnd:
            d['_mbEnd'] = mbEnd
            rhs_ads = rhs_ads + mbEnd.findAll('li')
        # rhs_ads from nobr
        nobr = d['_rhs_block'].find(name='ol',attrs={'class':'nobr'})
        if nobr:
            d['_nobr'] = nobr
            rhs_ads = rhs_ads + nobr.findAll('li')
            # puting rhs_ads in the dict
        if rhs_ads:
            d['_rhs_ads_list'] = rhs_ads
    # Okay, no more parsing wishes, return the dict d
    return d


def parse_tag_dict(tag_dict):
    d = {
        'number_of_results':None,
        'top_ads_list':[],
        'organic_results_list':[],
        'rhs_ads_list':[],
        'related_search_list':[]
    }
    if tag_dict.has_key('_top_ads_list'):
        for x in tag_dict['_top_ads_list']:
            xx = parse_ad(x)
            if xx: d['top_ads_list'].append(xx)
        # d['top_ads_list'] = [d['top_ads_list'].append(parse_ad(x)) for x in tag_dict['_top_ads_list'] if x!=None]
    if tag_dict.has_key('_organic_results_list'):
        for x in tag_dict['_organic_results_list']:
            xx = parse_organic_result(x)
            if xx: d['organic_results_list'].append(xx)
    if tag_dict.has_key('_rhs_ads_list'):
        for x in tag_dict['_rhs_ads_list']:
            xx = parse_ad(x)
            if xx: d['rhs_ads_list'].append(xx)
    if tag_dict.has_key('_related_search'):
        for x in tag_dict['_related_search']:
            xx = parse_related_search_list(x)
            if xx: d['related_search_list'].append(xx)

    #     organic_results_parsed = []
    #     for org_res in d['_organic_results_list']:
    #         org_result_parsed_item = parse_organic_result(org_res)
    #         if org_result_parsed_item:
    #             organic_results_parsed.append(org_result_parsed_item)
    #     parsed_d['organic_results_parsed'] = organic_results_parsed
    # if tag_dict.has_key('_rhs_ads_list'):
    #     rad_parsed = []
    #     for tad in d['_rhs_ads_list']:
    #         rad_parsed_item = parse_ad(tad)
    #         if rad_parsed_item:
    #             rad_parsed.append(parse_ad(rad_parsed_item))
    #     parsed_d['rad_parsed'] = rad_parsed
    # # Okay, no more parsing wishes, return the dict d
    # if d.has_key('_related_search'):

    return d



def parse_organic_result(search_ires_li_instance):
    d = dict()
    title_html = search_ires_li_instance.findAll(name='h3',attrs={'class':'r'})
    if title_html:
        title_html = title_html[-1] # take the last one only
        d = dict(d,**{'title_html':title_html})
        d = dict(d,**{'title_text':title_html.text}) # the text of the title

    div_class_s = search_ires_li_instance.findAll(name='div',attrs={'class':'s'})
    if div_class_s:
        div_class_s = div_class_s[-1] # take the last one only (hopefully the one just after the last class='r')
        cite_text = get_section(div_class_s,attrs={},name='cite')
        if cite_text:
            d = dict(d,**{'cite_text':cite_text.text})
            d = dict(d,**{'domain':tldextract.extract(cite_text.text).domain})

        f_slp_text = get_section(div_class_s,attrs={'class':'f slp'},name='div')
        if f_slp_text: d = dict(d,**{'f_slp_text':f_slp_text.text})

        st_text = get_section(div_class_s,attrs={'class':'st'},name='span')
        if st_text:
            st_text = st_text.text
            if st_text: d = dict(d,**{'st_text':st_text})

        osl = get_section(div_class_s,attrs={'class':'osl'},name='div')
        if osl: d = dict(d,**{'osl':osl})

        orp_table = get_section(div_class_s,attrs={},name='table')
        if orp_table:
            table_tds = orp_table.findAll('td')
            table_tds = [x for x in table_tds if x.text] # keep only the tds that have text
            d = dict(d,**{'table_tds':table_tds})
            # the following is specific to single hotel hits. May not be applicable to all orp_table.table_tds encountered
            if len(table_tds)>=1:
                td = table_tds[0]
                span = td.find(name='span')
                if span:
                    d = dict(d,**{'td0_span_text':span.get_text(separator=u"\n",strip=True)})
                fl = td.findAll(name='a',attrs={'class':'fl'})
                if fl:
                    d = dict(d,**{'td0_fl':[x.get_text(separator=u"\n",strip=True) for x in fl]})
            if len(table_tds)>=2:
                td1 = table_tds[1]
                if td1:
                    d = dict(d,**{'td1':td1.find(name='a').get('href')})
                    d = dict(d,**{'td1_text':td1.get_text(separator=u"\n",strip=True)})

    organic_result_type = 0 # will remain 0 if dict has no organic_results_parsed key
    if d:
        organic_result_type = 1 # default organic result type
        d = dict(d,**{'organic_result_type':1}) # default organic result type
        if d.has_key('table_tds'):
            if all([d.has_key(x) for x in ['td0_fl', 'td1_text']]):
                organic_result_type = 2 # a specific hotel google meta listing
            else:
                organic_result_type = 3 # something else with a table in it
    d = dict(d,**{'organic_result_type':organic_result_type})

    return d

def parse_ad(rad):
    d = dict()
    t = rad.find('h3').find('a')

    dest_url = t.get('href')
    if dest_url:
        d = dict(d,**{'dest_url':dest_url})
        dest_url_parsed = parse_qs(dest_url)
        if dest_url_parsed:
            dest_url_parsed = {k:v[0] for k,v in dest_url_parsed.iteritems()}
            if dest_url_parsed:
                d['dest_url_parsed'] = dest_url_parsed
                if dest_url_parsed.has_key('adurl'):
                    adurl = dest_url_parsed['adurl']
                    if adurl:
                        d['adurl'] = adurl
                        d['adurl_domain'] = tldextract.extract(adurl).domain

    title = t.getText()
    if title: d = dict(d,**{'title':title})

    disp_url = rad.find('div','kv')
    if disp_url:
        d['disp_url'] = disp_url.getText()
        d['disp_url_domain'] = tldextract.extract(d['disp_url']).domain
    #
    ad_text_html = rad.find('span','ac')
    if ad_text_html:
        d['ad_text_html'] = ad_text_html.renderContents()
        ad_text_lines = [re.sub(r"</?b>","",x) for x in d['ad_text_html'].split('<br/>')]
        if len(ad_text_lines)>=1:
            d['ad_text_line_1'] = ad_text_lines[0]
            if len(ad_text_lines)>=2:
                d['ad_text_line_2'] = ad_text_lines[1]
            else:
                d['ad_text_line_2'] = ''
        else:
            d['ad_text_line_1'] = ''


    div_f = rad.find('div','f')
    if div_f:
        d['div_f'] = div_f.renderContents()
        d['div_f_text'] = div_f.get_text('|||')

    # ad_text = ttt.getText(separator='|||')
    return d

def parse_related_search_list(rel_search_tag):
    d = dict()
    rel_search_list = rel_search_tag.findAll('td')
    if rel_search_list:
        return [x.get_text() for x in rel_search_list]
    else:
        return []


#
# def mk_result_dict(input):
#     """
#     mk_result_dict(input)
#     takes a soup, html string, or filename of google result htmls as an input
#     and returns a dict containing components of the html we're interested in
#     """
#     input = util.x_to_soup(input)
#     d = dict()
#
#     # number of results
#     resultStats = extract_section(input,attrs={'id':'resultStats'},name='div')
#     if resultStats: d = dict(d,**{'resultStats':resultStats})
#
#     # center_col
#     center_col = extract_section(input,attrs={'id':'center_col'},name='div')
#     if center_col:
#         d = dict(d,**{'center_col':center_col})
#         # tads
#         tads = extract_section(d['center_col'],attrs={'id':'tads'},name='div')
#         if tads:
#             d = dict(d,**{'tads':tads})
#             # top_ads
#             top_ads = d['tads'].findAll('li')
#             if top_ads:
#                 d = dict(d,**{'top_ads':top_ads})
#         # res
#         res = extract_section(d['center_col'],attrs={'id':'res'},name='div')
#         if res:
#             d = dict(d,**{'res':res})
#             # searchInstead
#             topstuff = extract_section(d['res'],attrs={'id':'topstuff'},name='div')
#             if topstuff:
#                 d = dict(d,**{'topstuff':topstuff})
#                 # spell
#                 spell = extract_section(d['topstuff'],attrs={'class':'spell'},name='a')
#                 if spell: d = dict(d,**{'spell':spell})
#             # search
#             search = extract_section(d['res'],attrs={'id':'search'},name='div')
#             if search:
#                 d = dict(d,**{'search':search})
#                 # ires
#                 ires = extract_section(d['search'],attrs={'id':'ires'},name='div')
#                 if ires:
#                     d = dict(d,**{'ires':ires})
#                     # organicResults
#                     organic_results = d['ires'].findAll('li')
#                     if organic_results:
#                         d = dict(d,**{'organic_results':organic_results})
#                         organic_results_parsed = []
#                         for org_res in d['organic_results']:
#                             org_result_parsed_item = organic_result_parser(org_res)
#                             if org_result_parsed_item:
#                                 organic_results_parsed.append(org_result_parsed_item)
#                         if organic_results_parsed:
#                             d = dict(d,**{'organic_results_parsed':organic_results_parsed})
#         # related_search
#         t = [x.find(name='table') for x in d['center_col'].findAll(name='div')]
#         t = [xx for xx in t if xx!=None]
#         if t:
#             related_search = t[0].findAll('td')
#             d = dict(d,**{'related_search':related_search})
#
#     # rhs_block
#     rhs_block = extract_section(input,attrs={'id':'rhs_block'},name='div')
#     if rhs_block:
#         d = dict(d,**{'rhs_block':rhs_block})
#         rhs_ads = [] # initializing
#         # rhs_ads from mbEnd
#         mbEnd = extract_section(d['rhs_block'],attrs={'id':'mbEnd'},name='div')
#         if mbEnd:
#             d = dict(d,**{'mbEnd':mbEnd})
#             rhs_ads = rhs_ads + mbEnd.findAll('li')
#         # rhs_ads from nobr
#         nobr = extract_section(d['rhs_block'],attrs={'class':'nobr'},name='ol')
#         if nobr:
#             d = dict(d,**{'nobr':nobr})
#             rhs_ads = rhs_ads + nobr.findAll('li')
#             # puting rhs_ads in the dict
#         if rhs_ads:
#             d = dict(d,**{'rhs_ads':rhs_ads})
#     # Okay, no more parsing wishes, return the dict d
#     return d



# # below is the first version of organic_result_parser. It assumed there'd be only one class='r', but I found multiple in some special results

# def organic_result_parser(input):
#     input = x_to_soup(input)
#     d = dict()
#     title_html = get_section(input,attrs={'class':'r'},name='h3')
#     if title_html:
#         d = dict(d,**{'title_html':title_html})
#         d = dict(d,**{'title_text':title_html.text})
#
#     div_class_s = get_section(input,attrs={'class':'s'},name='div')
#     if div_class_s:
#         cite_text = get_section(div_class_s,attrs={},name='cite')
#         if cite_text: d = dict(d,**{'cite_text':cite_text.text})
#
#         f_slp_text = get_section(div_class_s,attrs={'class':'f slp'},name='div')
#         if f_slp_text: d = dict(d,**{'f_slp_text':f_slp_text.text})
#
#         st_text = get_section(div_class_s,attrs={'class':'st'},name='span')
#         if st_text: d = dict(d,**{'st_text':st_text.text})
#
#         osl = get_section(div_class_s,attrs={'class':'osl'},name='div')
#         if osl: d = dict(d,**{'osl':osl})
#
#         tbody = get_section(div_class_s,attrs={},name='tbody')
#         if tbody:
#             tbody_tds = tbody.findAll('td')
#             d = dict(d,**{'tbody_tds':tbody_tds})
#
#     return d




##################################################################################
# UTILS
##################################################################################




##################################################################################
# OTHER FUNCTIONS
# TODO: move or remove (don't think we'll need these any more)
# TODO: But before (re)moving, consider the logic in these functions: May want to reuse some of it
##################################################################################


def mk_rhs_ads_dict(li_tag):
    d =  {
        'h3':extract_section(li_tag,attrs={'id':'taw'},name='h3'),
        'kv':extract_section(li_tag,attrs={'class':'kv'},name='div'),
        'f':extract_section(li_tag,attrs={'class':'f'},name='div'),
        'extrares':extract_section(li_tag,attrs={'class':'ac'},name='span')
    }
    return {i:j for i,j in d.items() if j != None}

def expand_node(node,expand_def):
    if isinstance(expand_def,str):
        return extract_section(node,attrs={'id':expand_def},name='div')
    elif isinstance(expand_def,dict) and expand_def.has_key('attrs'):
        if expand_def.has_key('name'):
            return extract_section(node,expand_def['attrs'],name=expand_def['name'])
        else:
            return extract_section(node,expand_def['attrs'],name='div')
    # TODO: exception throwing

def rm_empty_dict_values(d):
    for k in d.keys():
        if not d[k]:
            d.pop(k)
    # TODO: test if the following dict comprehension accelerates things
    # TODO: (this approach returns a modified dict instead of changing things in place
    # return {i:j for i,j in d.items() if j != None}

def mk_tag_dict(soup):
    d = root_dict(soup)
    to_expand = ['center_col','rhscol','taw','res','extrares','rhscol']
    for t in to_expand:
        if d.has_key(t): d = dict(d,**extract_tag_dict_from_node(d[t],t))

def root_dict(soup):
    soup = util.x_to_soup(soup)
    return extract_tag_dict_from_node(soup,'root')

def extract_tag_dict_from_node(node, dict_spec):
    if dict_spec=='root':
        appbar = extract_section(node,attrs={'id':'appbar'},name='div')
        rcnt = extract_section(node,attrs={'id':'rcnt'},name='div')
        center_col = extract_section(rcnt,attrs={'id':'center_col'},name='div')
        rhscol = extract_section(rcnt,attrs={'id':'rhscol'},name='div')
        d = {
            'appbar':appbar,
            'center_col':center_col,
            'rhscol':rhscol
        }
    elif dict_spec=='rhs_ads':
        d = {
            'h3':extract_section(node,attrs={'id':'taw'},name='h3'),
            'kv':extract_section(node,attrs={'class':'kv'},name='div'),
            'f':extract_section(node,attrs={'class':'f'},name='div'),
            'extrares':extract_section(node,attrs={'class':'ac'},name='span')
        }
    elif dict_spec=='center_col':
        d = {
            'taw':extract_section(node,attrs={'id':'taw'},name='span'),
            'res':extract_section(node,attrs={'id':'res'},name='div'),
            'extrares':extract_section(node,attrs={'id':'extrares'},name='div')
        }
    elif dict_spec=='taw':
        d = {
            'tads':extract_section(node,attrs={'id':'tads'},name='div'), # top ads
            'cu':extract_section(node,attrs={'id':'cu'},name='div') # hotel finder
        }
    elif dict_spec=='ires':
        d = {
            'search':extract_section(node,attrs={'id':'search'},name='div') # organic search results
        }
    elif dict_spec=='extrares':
        d = {
            'brs':extract_section(node,attrs={'id':'brs'},name='div') # related searches
        }
    elif dict_spec=='rhs_block':
        d = {
            'mbEnd':extract_section(node,attrs={'id':'brs'},name='div'), # side ads
            'knop':extract_section(node,attrs={'id':'knop'},name='div'),
            'rhsvw vk_rhsc':extract_section(node,attrs={'id':'rhsvw vk_rhsc'},name='div') # hotel finder (specific)
        }
    else:
        d = {}
    d = {i:j for i,j in d.items() if j != None} # remove keys with empty values
    return d



def get_section(soup, attrs={}, name='div', all=False):
    """
    gets the bs4.element.Tag (or list thereof) of a section specified by the attrs (dict or list of dicts)
    """
    if all==False:
        if isinstance(attrs,dict):
            return soup.find(name=name, attrs=attrs)
        else:
            tag = soup
            for ss in attrs:
                tag = tag.find(name=name, attrs=ss)
            return tag
    else:
        if isinstance(attrs,dict):
            return soup.findAll(name=name, attrs=attrs)
        else: # not sure how to handle this, so I'm forcing exit
            print "haven't coded this yet"
            return None


def num_of_results(input):
    """
    returns the number of google results.
    Function assumes it will be the number just before the last "result" match of the appbar div (sometimes there's
    other results (like "personal results")
    """
    return num_of_results_soup01(input)

def top_elements(input):
    """
    the top ads, hotel finder and other elements coming before webResults
    """
    input = util.x_to_soup(input)
    return input.findAll('span', attrs={'id':'taw'})
    # taw = input.findAll('span', attrs={'id':'taw'})
    # top_ads = taw[0].findAll('div', attrs={'id':'tads'})
    # pre_results = taw[0].findAll('div', attrs={'class':'c'})
    # return dict({'top_ads':top_ads},**{'pre_results':pre_results})

def google_webResults(input):
    """
    google_webResults
    """
    return google_webResults_soup(input)

def rh_ads(input):
    """
    right hand side ads
    """
    output = []
    input = util.x_to_soup(input)
    rhs_block_element = rhs_block(input)
    if len(rhs_block_element)!=0:
        ol_section = rhs_block_element[0].findAll('ol')
        if len(ol_section):
            return [str(li) for li in ol_section[0].findAll('li')] # TODO: do we want string, unicode, or the tag itself as is?
    # else returns empty list

####################################
# MENU OF POSSIBLE PARSING FUNCTIONS


def num_of_results_lxml(file_spec):
    """
    num_of_results using  lxml
    """
    file_spec = util.x_to_file(file_spec)
    doc = etree.parse(file_spec, etree.HTMLParser())
    t = doc.xpath(XP_NRESULT)
    # return int(CRE_NUMBER.search(t[-1]).group(0).replace(',',''))
    return int(re.sub(RE_NUM_SEP,"",CRE_NUMBER.search(t[-1]).group(0)))

def num_of_results_soup01(soup):
    """
    num_of_results using Beautifulsoup
    """
    soup = util.x_to_soup(soup)
    assert isinstance(soup,BeautifulSoup), "hey, I'm expecting soup!"
    t = get_section(soup,[{'id':'appbar'},{'id':'topabar'}])
    t = re.findall("("+RE_NUMBER+").*?result",t.text)
    return int(re.sub(RE_NUM_SEP,"",t[-1]))

def google_webResults_soup(source):
    """
    google_webResults using Beautifulsoup
    """
    soup = util.x_to_soup(source)
    list_of_li = soup.findAll('li', attrs={'class':'g'})
    return [
        {'pos':i+1, 'title': li.find('a'), 'text': li.find('span', attrs={'class':'st'})}
        for (i,li) in enumerate(list_of_li)
    ]


##################################################################################
# Parsing sections from soup (returns soup bs4.element.Tag objects)
##################################################################################

########### Level 1

def topabar(soup):
    return soup.findAll('div', attrs={'id':'topabar'})

def tvcap(soup):
    return soup.findAll('div', attrs={'id':'tvcap'})

def res__main(soup):
    return soup.findAll('div', attrs={'id':'res','role':'main'})

def extrares(soup):
    return soup.findAll('div', attrs={'id':'extrares'})

def rhs_block(soup):
    return soup.findAll('div', attrs={'id':'rhs_block'})





if __name__ == "__main__":
    print "you just ran ut.parse.google"
    # is_none = lambda x : [xx==None for xx in x]
    # index_of_trues = lambda x : [idx for idx in range(len(x)) if x[idx]==True]
    # idx_of_nones_of = lambda x : index_of_trues(is_none(x))
    # folder = '/D/Dropbox/dev/py/data/tmp'
    # file_list = ['www hidden camera in hotel com.html',
    #              'beekmann tower hotel in new york.html',
    #              'campo prague hotel.html',
    #              'b&b romantica venezia.html',
    #              '5 star hotels in salta argentina.html',
    #              '$100 hotel in amsterdam.html',
    #              '1 star hotel.html',
    #              'albergo studi vernazza.html',
    #              'asport.html']
    # #file_list = os.listdir(folder)
    # #file_list = filter ((lambda s: re.search(r"\.html$", s)), file_list)
    # filepath_list = [os.path.join(folder,f) for f in file_list]
    # my_parser = lambda f: mk_result_dict(f)

    # # test if the parser returned any Nones for the filepath_list
    # r = [my_parser(f) for f in filepath_list]
    # idx = idx_of_nones_of(r)
    # if len(idx)==0:
    #     print "all {} htmls had some data".format(len(filepath_list))
    # else:
    #     print "{}/{} htmls didn't have any parsed data".format(len(idx),len(filepath_list))
    #     print "the indices of these are in the variable idx"
    #
    # # Check what keys were parse
    # key_list = ['center_col', 'ires', 'mbEnd', 'nobr', 'organic_results', 'related_search', 'res', 'resultStats', 'rhs_ads', 'rhs_block', 'search', 'spell', 'tads', 'top_ads', 'topstuff']
    # idx_of_items_not_having_key = dict()
    # for key in key_list:
    #     t = [x[key] for x in r if x.has_key(key)]
    #     lidx = [x.has_key(key) for x in r]
    #     idx = [i for i,x in enumerate(lidx) if x==False]
    #     idx_of_items_not_having_key = dict(idx_of_items_not_having_key,**{key : [i for i,x in enumerate(lidx) if x==False]})
    #     print "{}/{} have a {}".format(len(t),len(r),key)
    # print ""
    # print "--> Indices of items not having specific keys may be found in the dict: idx_of_items_not_having_key"
    # files_not_having_key = lambda key : [filepath_list[i] for i in idx_of_items_not_having_key[key]]
    # print "--> Filenames of items not having specific keys may be found with the function: files_not_having_key(key)"
    #
    # # see files with missing key...
    # key = 'rhs_ads'
    # files_not_having_key('rhs_ads')


