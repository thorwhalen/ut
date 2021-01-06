"""Parsing google results"""
__author__ = 'thorwhalen'

#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import os, sys
from bs4 import BeautifulSoup
import re
import os.path
from ut.pfile.name import files_of_folder
from ut.pfile.name import replace_folder_and_ext
from urllib.request import urlopen
from ut.pfile import to
from lxml import etree
import pickle
import tldextract
import ut.parse.util as putil
from urllib.parse import urlparse, parse_qs

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



def get_info_dict_01(input):
    input = putil.x_to_soup(input)


# def filter_info_dict_01(info_dict):
#     if info_dict.haskey('organic_results_list'):
#         t = 2
#         # TODO: Continue coding
#     # info_dict = {k:info_dict[k] for k in ('number_of_results','',''}
#

def get_info_dict_debug(input):
    # input handling
    if not isinstance(input,dict):
        if isinstance(input,str):
            if os.path.isfile(input):
                input = BeautifulSoup(input)
            else:
                input = mk_gresult_tag_dict(input)
        if isinstance(input,BeautifulSoup):
            input = mk_gresult_tag_dict(input)
    return input


def get_info_dict(input):
    # input handling
    if not isinstance(input,dict):
        if isinstance(input,str):
            if os.path.isfile(input):
                input = BeautifulSoup(input)
            else:
                input = mk_gresult_tag_dict(input)
        if isinstance(input,BeautifulSoup):
            input = mk_gresult_tag_dict(input)
    # at this point we assume input is a tag_dict
    # get the info dict
    return parse_tag_dict(input)

def mk_gresult_tag_dict(input):
    """
    mk_result_dict(input)
    takes a soup, html string, or filename of google result htmls as an input
    and returns a dict containing components of the html we're interested in
    """
    input = putil.x_to_soup(input)
    d = dict()

    # number of results
    resultStats = input.find(name='div',attrs={'id':'resultStats'})
    if resultStats: d['_resultStats'] = resultStats

    # center_col
    center_col = input.find(name='div',attrs={'id':'center_col'})
    if center_col:
        d['_center_col'] = center_col
        # tads   (center_col.taw.tvcap.tads)
        tads = d['_center_col'].find(name='div',attrs={'id':'tads'})
        if tads:
            d['_tads'] = tads
            # top_ads
            top_ads = d['_tads'].findAll('li')
            if top_ads:
                d['_top_ads_list'] = top_ads
        # c   (center_col.taw.tvcap.c)
        c = d['_center_col'].find(name='div',attrs={'class':'c'})
        if c:
            d['_c'] = c
            # c_list
            c_list = d['_c'].findAll('li')
            if c_list:
                d['_c_list'] = c_list
        # res
        res = d['_center_col'].find(name='div',attrs={'id':'res'})
        if res:
            d['_res'] = res
            # searchInstead
            topstuff = d['_res'].find(name='div',attrs={'id':'topstuff'})
            if topstuff:
                d['_topstuff'] = topstuff # used to contain spell,
                # but then realized the spell appeared in other places some times, so moved it to child of _center_col
            # spell
            spell = d['_center_col'].find(name='a',attrs={'class':'spell'})
            if spell:
                d['_spell'] = spell
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

        # # related_search
        # extrares = d['_center_col'].find(name='div',attrs={'id':'extrares'})
        # if extrares:
        #     related_search = extrares.find('table')
        #     if related_search:
        #         d['_related_search'] = related_search

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
        # lu_pinned_rhs (where some hotel finder, maps, specific hotels might be)
        lu_pinned_rhs = d['_rhs_block'].find(name='div',attrs={'id':'lu_pinned_rhs'})
        if lu_pinned_rhs:
            d['_lu_pinned_rhs'] = lu_pinned_rhs
        # knop (another place where some hotel finder, maps, specific hotels might be)
        knop = d['_rhs_block'].find(name='div',attrs={'id':'knop'})
        if knop:
            d['_knop'] = knop
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
    if '_resultStats' in tag_dict:
        xx = parse_number_of_results(tag_dict['_resultStats'])
        if xx: d['number_of_results'] = xx
    if '_top_ads_list' in tag_dict:
        for x in tag_dict['_top_ads_list']:
            xx = parse_ad(x)
            if xx: d['top_ads_list'].append(xx)
        # d['top_ads_list'] = [d['top_ads_list'].append(parse_ad(x)) for x in tag_dict['_top_ads_list'] if x!=None]
    if '_organic_results_list' in tag_dict:
        for x in tag_dict['_organic_results_list']:
            xx = parse_organic_result(x)
            if xx: d['organic_results_list'].append(xx)
    if '_rhs_ads_list' in tag_dict:
        for x in tag_dict['_rhs_ads_list']:
            xx = parse_ad(x)
            if xx: d['rhs_ads_list'].append(xx)
    if '_related_search' in tag_dict:
        for x in tag_dict['_related_search']:
            xx = parse_related_search_list(x)
            if xx: d['related_search_list'].append(xx)
    return d

def parse_number_of_results(resultStats_tag):
    t = re.findall("("+RE_NUMBER+").*?result",resultStats_tag.text)
    if t:
        return int(re.sub(RE_NUM_SEP,"",t[-1]))
    else:
        return None

def parse_organic_result(search_ires_li_instance):
    d = dict()
    title_html = search_ires_li_instance.findAll(name='h3',attrs={'class':'r'})
    if title_html:
        title_html = title_html[-1] # take the last one only
        d['title_text'] = title_html.text

    div_class_s = search_ires_li_instance.findAll(name='div',attrs={'class':'s'})
    if div_class_s:
        div_class_s = div_class_s[-1] # take the last one only (hopefully the one just after the last class='r')
        cite_text = div_class_s.find(name='cite',attrs={})
        if cite_text:
            d['cite_text'] = cite_text.text # link attached to the title of the organic result
            d['domain'] = tldextract.extract(d['cite_text'].split(' ')[0]).domain # domain of this link

        f_slp_text = div_class_s.find(name='div',attrs={'class':'f slp'})
        if f_slp_text: d['f_slp_text'] = f_slp_text.text

        st_text = div_class_s.find(name='span',attrs={'class':'st'})
        if st_text:
            st_text = st_text.text
            if st_text: d['st_text'] = st_text

        osl = div_class_s.find(name='div',attrs={'class':'osl'})
        if osl: d['osl'] = osl.renderContents()

        _orp_table = div_class_s.find(name='table',attrs={})
        if _orp_table:
            table_tds_list = _orp_table.findAll('td')
            table_tds_list = [x for x in table_tds_list if x.text] # keep only the tds that have text
            d['table_tds_list'] = [x.renderContents() for x in table_tds_list]
            # the following is specific to single hotel hits. May not be applicable to all _orp_table.table_tds_list encountered
            if len(table_tds_list)>=1:
                td = table_tds_list[0]
                span = td.find(name='span')
                if span:
                    d['td0_span_text'] = span.get_text(separator="\n",strip=True)
                    # d = dict(d,**{'td0_span_text':span.get_text(separator=u"\n",strip=True)})
                fl = td.findAll(name='a',attrs={'class':'fl'})
                if fl:
                    d['td0_fl'] = [x.get_text(separator="\n",strip=True) for x in fl]
                    # d = dict(d,**{'td0_fl':[x.get_text(separator=u"\n",strip=True) for x in fl]})
            if len(table_tds_list)>=2:
                td1 = table_tds_list[1]
                if td1:
                    td1_href = td1.find(name='a')
                    if td1_href:
                        d['td1'] = td1_href.get('href')
                        d['td1_text'] = td1.get_text(separator="\n",strip=True)
                    #d['td1'] = td1.find(name='a').get('href')
                    #d['td1_text'] = td1.get_text(separator=u"\n",strip=True)

    organic_result_type = 0 # will remain 0 if dict has no organic_results_parsed key
    if d:
        organic_result_type = 1 # default organic result type
        d['organic_result_type'] = 1 # default organic result type
        # d = dict(d,**{'organic_result_type':1}) # default organic result type
        if 'table_tds_list' in d:
            if all([x in d for x in ['td0_fl', 'td1_text']]):
                organic_result_type = 2 # a specific hotel google meta listing
            else:
                organic_result_type = 3 # something else with a table in it
    d['organic_result_type'] = organic_result_type
    # d = dict(d,**{'organic_result_type':organic_result_type})

    return d

def parse_ad(rad):
    d = dict()
    t = rad.find('h3').find('a')

    dest_url = t.get('href')
    if dest_url:
        d['dest_url'] = dest_url
        # d = dict(d,**{'dest_url':dest_url})
        dest_url_parsed = parse_qs(dest_url)
        if dest_url_parsed:
            dest_url_parsed = {k:v[0] for k,v in dest_url_parsed.items()}
            if dest_url_parsed:
                d['dest_url_parsed'] = dest_url_parsed
                if 'adurl' in dest_url_parsed:
                    adurl = dest_url_parsed['adurl']
                    if adurl:
                        d['adurl'] = adurl
                        d['adurl_domain'] = tldextract.extract(adurl).domain

    title = t.getText()
    if title:
        d['title'] = title
        #d = dict(d,**{'title':title})

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


    div_f_html = rad.find('div','f')
    if div_f_html:
        d['div_f_html'] = div_f_html.renderContents()
        d['div_f_text'] = div_f_html.get_text('|||')

    # ad_text = ttt.getText(separator='|||')
    return d

def parse_related_search_list(rel_search_tag):
    d = dict()
    rel_search_list = rel_search_tag.findAll('td')
    if rel_search_list:
        return [x.get_text() for x in rel_search_list]
    else:
        return []

def get_domain_list_from_google_results(gresults):
    domain_list = []
    if not isinstance(gresults,dict): # assume it's a soup, html, or filename thereof
        gresults = parse_tag_dict(mk_gresult_tag_dict(gresults))
    # if not, assume the input is a info_dict
    if 'organic_results_list' in gresults:
        domain_list = domain_list + [x['domain'] for x in gresults['organic_results_list'] if 'domain' in x]
    if 'top_ads_list' in gresults:
        domain_list = domain_list + [x['disp_url_domain'] for x in gresults['top_ads_list'] if 'disp_url_domain' in x]
    if 'organic_results_list' in gresults:
        domain_list = domain_list + [x['disp_url_domain'] for x in gresults['organic_results_list'] if 'disp_url_domain' in x]
    return domain_list

##################################################################################
# DIAGNOSIS
##################################################################################

def element_presence_dict(input):
    input = putil.x_to_soup(input)
    html = input.renderContents()
    text = input.get_text(' ')
    d = dict()
    # search_term_redirect (when google shows results for a different search term than the entered--should be in _spell)
    d['isa_search_term_redirect'] = bool(re.search('(Showing results for)|(Including results for)',text))
    # did_you_mean
    d['did_you_mean'] = bool(re.search('Did you mean',text))
    # number of results
    d['number_of_results'] = re.findall('About ('+RE_NUMBER+') results',text)
    if len(d['number_of_results']) > 0:
        d['isa_number_of_results'] = True
    else:
        if bool(re.search('Your search .* did not match any documents.')):
            d['number_of_results'] = 0
        else:
            d['number_of_results'] = -1
    # number of center_col_elements
    center_col = input.find('div',{'id':'center_col'})
    if center_col:
        d['num_of_center_col_children'] = len([x for x in center_col.children])
    else:
        d['num_of_center_col_children'] = 0

    # Showing results for



##################################################################################
# UTILS
##################################################################################

def html_to_info_dicts(source,tag_folder,info_folder,printProgress=True):
    if isinstance(source,str) and os.path.isdir(source):
        source_folder = source
        source = [os.path.join(source_folder,f) for f in files_of_folder(source)]
    else:
        assert isinstance(source,list),"source must be a folder or a list of filepaths"
    for i,f in enumerate(source):
        if printProgress==True:
            print("{}: {}".format(i,f))
        d = mk_gresult_tag_dict(f)
        pickle.dump(d,open(replace_folder_and_ext(f,tag_folder,'tag_dict'),'w'))
        d = parse_tag_dict(d)
        pickle.dump(d,open(replace_folder_and_ext(f,info_folder,'info_dict'),'w'))


##################################################################################
# OTHER FUNCTIONS
# TODO: move or remove (don't think we'll need these any more)
# TODO: But before (re)moving, consider the logic in these functions: May want to reuse some of it
##################################################################################




if __name__ == "__main__":

    print("you just ran ut.parse.google")
    html_folder = '/D/Dropbox/dev/py/data/html/google_results_tests/'
    tag_folder = '/D/Dropbox/dev/py/data/dict/google_results_tag_dict/'
    info_folder = '/D/Dropbox/dev/py/data/dict/google_results_info_dict/'
    file_list = ['www hidden camera in hotel com.html',
                 'beekmann tower hotel in new york.html',
                 'campo prague hotel.html',
                 'b&b romantica venezia.html',
                 '5 star hotels in salta argentina.html',
                 '5 star hotels in salta argentina - Google Search.html',
                 'hotels in paris - Google Search.html',
                 'hotel ducs de bourgogne - Google Search.html',
                 '$100 hotel in amsterdam.html',
                 '1 star hotel.html',
                 'albergo studi vernazza.html',
                'asport.html',
                'beautifulsoup difference between getText and get_text.html']
#file_list = os.listdir(html_folder)
#file_list = filter ((lambda s: re.search(r"\.html$", s)), file_list)

    #filepath_list = [os.path.join(html_folder,f) for f in file_list]
    #html_to_info_dicts(filepath_list[:3],tag_folder,info_folder,printProgress=True)

    # d = mk_gresult_tag_dict(os.path.join(html_folder,file_list[1]))
    # dd = parse_tag_dict(d)
    # from pdict import diagnosis
    # diagnosis.dict_of_types_of_dict_values(dd,recursive=True)


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


