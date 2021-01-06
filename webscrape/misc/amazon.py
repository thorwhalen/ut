"""Scraping and parsing amazon"""
__author__ = 'thor'

import os
from ut.util.importing import get_environment_variable
import ut as ms
import ut.dacc.mong.util
import pandas as pd
import numpy as np
import requests
import re
from BeautifulSoup import BeautifulSoup as bs3_BeautifulSoup
from datetime import timedelta
from datetime import datetime
from pymongo import MongoClient
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from ut.serialize.s3 import S3
import tempfile
from ut.viz.util import insert_nans_in_x_and_y_when_there_is_a_gap_in_x
import pylab


class Amazon(object):
    url_template = dict()
    url_template['product_page'] = 'http://www.amazon.{country}/dp/{asin}/'
    url_template['product_reviews'] = 'http://www.amazon.{country}/product-reviews/{asin}/'

    regexp = dict()
    regexp['nreviews_re'] = {'com': re.compile('\d[\d,]*(?= customer review)'),
                             'co.uk': re.compile('\d[\d,]*(?= customer review)'),
                             'in': re.compile('\d[\d,]*(?= customer review)'),
                             'de': re.compile('\d[\d\.]*(?= Kundenrezens\w\w)')}
    regexp['no_reviews_re'] = {'com': re.compile('no customer reviews'),
                               'co.uk': re.compile('no customer reviews'),
                               'in': re.compile('no customer reviews'),
                               'de': re.compile('Noch keine Kundenrezensionen')}
    # regexp['average_rating_re'] = {'com': re.compile('')}
    default = dict()
    default['country'] = 'com'
    # default['requests_kwargs'] = {}
    default['requests_kwargs'] = {
        'proxies': {'http': 'http://us.proxymesh.com:31280'},
        'auth': requests.auth.HTTPProxyAuth(get_environment_variable('PROXYMESH_USER'),
                                            get_environment_variable('PROXYMESH_PASS'))
    }

    @classmethod
    def url(cls, what='product_page', **kwargs):
        kwargs = dict(Amazon.default, **kwargs)
        return cls.url_template[what].format(**kwargs)
        return r.text

    @classmethod
    def slurp(cls, what='product_page', **kwargs):
        kwargs = dict(Amazon.default, **kwargs)
        r = requests.get(Amazon.url(what=what, **kwargs), **Amazon.default['requests_kwargs'])
        if r.status_code == 200:
            return r.text
        else:  # try again and return no matter what
            r = requests.get(Amazon.url(what=what, **kwargs), **Amazon.default['requests_kwargs'])
            return r.text

    # @classmethod
    # def get_dynamic_book_info(cls, asin, **kwargs):
    #     html = Amazon.slurp(what='product_page', **kwargs)
    #     b = bs3_BeautifulSoup(b)


    @classmethod
    def get_info(cls, asin, country='co.uk', **kwargs):
        info = {'date': datetime.now()}
        info = dict(info, **{'sales_ranks': cls.get_sales_rank(asin, country='co.uk', **kwargs)})
        # info = dict(info, **{'num_of_reviews': cls.get_number_of_reviews(asin, country='co.uk', **kwargs)})
        return info

    @classmethod
    def get_sales_rank(cls, **kwargs):
        html = Amazon.slurp(what='product_page', **kwargs)
        sales_rank = [Amazon.parse_sales_rank(html, **kwargs)]
        sales_rank += Amazon.parse_sales_sub_rank(html, **kwargs)
        return sales_rank

    @classmethod
    def parse_product_title(cls, b, **kwargs):
        if not isinstance(b, bs3_BeautifulSoup):
            b = bs3_BeautifulSoup(b)
        return b.find('span', attrs={'id': 'productTitle'}).text

    @classmethod
    def parse_sales_rank(cls, b, **kwargs):
        if not isinstance(b, bs3_BeautifulSoup):
            b = bs3_BeautifulSoup(b)
        t = b.find('li', attrs={'id': re.compile('SalesRank')})
        sales_rank_re = re.compile('(\d[\d,]+) in ([\w\ ]+)')
        tt = sales_rank_re.findall(t.text)
        return {'sales_rank': int(re.compile('\D').sub('', tt[0][0])),
                'sales_rank_category': tt[0][1].strip(' ')}

    @classmethod
    def parse_sales_sub_rank(cls, b, **kwargs):
        if not isinstance(b, bs3_BeautifulSoup):
            b = bs3_BeautifulSoup(b)
        t = b.find('li', attrs={'id': re.compile('SalesRank')})
        tt = t.findAll('li', 'zg_hrsr_item')
        sales_sub_rank = list()
        for tti in tt:
            d = dict()
            d['sales_rank'] = int(re.compile('\D').sub('', tti.find('span', 'zg_hrsr_rank').text))
            ttt = tti.find('span', 'zg_hrsr_ladder')
            ttt = ttt.text.split('&nbsp;')[1]
            d['sales_rank_category'] = ttt.split('&gt;')
            sales_sub_rank.append(d)
        return sales_sub_rank

    @classmethod
    def parse_avg_rating(cls, b, **kwargs):
        if not isinstance(b, bs3_BeautifulSoup):
            b = bs3_BeautifulSoup(b)
        t = b.find('span', 'reviewCountTextLinkedHistogram')
        return float(re.compile('[\d\.]+').findall(t['title'])[0])

    @classmethod
    def parse_product_title(cls, b, **kwargs):
        if not isinstance(b, bs3_BeautifulSoup):
            b = bs3_BeautifulSoup(b)
        t = b.find('div', attrs={'id': 'title'})
        return t.find('span', attrs={'id': 'productTitle'}).text

    @staticmethod
    def test_rating_scrape_with_vanessas_book():
        html = Amazon.slurp(what='product_page', country_ext='.co.uk', asin='1857886127')

    @staticmethod
    def get_number_of_reviews(asin, country, **kwargs):
        url = 'http://www.amazon.{country}/product-reviews/{asin}'.format(country=country, asin=asin)
        html = requests.get(url).text
        try:
            return int(re.compile('\D').sub('', Amazon.regexp['nreviews_re'][country].search(html).group(0)))
        except Exception:
            if Amazon.regexp['no_reviews_re'][country].search(html):
                return 0
            else:
                return None  # to distinguish from 0, and handle more cases if necessary


class AmazonBookWatch(object):
    default = dict()
    default['product_list'] = [
        {'title': 'The Nanologues', 'asin': '9350095173'},
        {'title': 'Never mind the bullocks', 'asin': '1857886127'},
        {'title': 'The Other Side of Paradise', 'asin': '1580055311'}
    ]
    default['watch_list'] = [
        {'title': 'The Nanologues', 'asin': '9350095173', 'country': 'in'},
        {'title': 'The Nanologues', 'asin': '9350095173', 'country': 'co.uk'},
        {'title': 'The Nanologues', 'asin': '9350095173', 'country': 'com'},
        {'title': 'Never mind the bullocks', 'asin': '1857886127', 'country': 'in'},
        {'title': 'Never mind the bullocks', 'asin': '1857886127', 'country': 'co.uk'},
        {'title': 'Never mind the bullocks', 'asin': '1857886127', 'country': 'com'},
        {'title': 'The Other Side of Paradise', 'asin': '1580055311', 'country': 'com'},
        {'title': 'The Other Side of Paradise', 'asin': '1580055311', 'country': 'co.uk'},
        {'title': "Heaven's Harlots (Paperback)", 'asin': '0688170129', 'country': 'com'},
        {'title': "Heaven's Harlots (Hardcover)", 'asin': '0688155049', 'country': 'com'},
        {'title': "Women on Ice", 'asin': '0813554594', 'country': 'com'}
    ]
    default['frequency_in_hours'] = 1
    default['max_date_ticks'] = 200
    default['stats_num_of_days'] = 1
    default['figheight'] = 3
    default['figwidth'] = 14
    default['linewidth'] = 3
    default['tick_font_size'] = 13
    default['label_fontsize'] = 13
    default['title_fontsize'] = 15
    default['line_style'] = '-bo'
    default['facecolor'] = 'blue'
    default['save_format'] = 'png'
    default['dpi'] = 40
    default['book_info_html_template'] = '''<hr>
        <h3>{book_title} - {country} - {num_of_reviews} reviews </h3>
    '''
    default['category_html'] = '<img style="box-shadow:         3px 3px 5px 6px #ccc;" src={image_url}>'
    db = MongoClient()['misc']['book_watch']

    def __init__(self, **kwargs):
        self.s3 = S3(bucket_name='public-ut-images', access_key='ut')
        attribute_name = 'product_list'
        setattr(self, attribute_name, kwargs.get(attribute_name, None) or AmazonBookWatch.default[attribute_name])
        attribute_name = 'watch_list'
        setattr(self, attribute_name, kwargs.get(attribute_name, None) or AmazonBookWatch.default[attribute_name])

    def asin_of_title(self, title):
        the_map = {k: v for k, v in zip([x['title'] for x in self.product_list], [x['asin'] for x in self.product_list])}
        return the_map[title]

    def get_book_statuses(self):
        now = datetime.now()
        info_list = list()
        for book in self.watch_list:
            try:
                info = dict({'date': now}, **book)
                info = dict(info, **{'sale_ranks': Amazon.get_sales_rank(**book)})
                info = dict(info, **{'num_of_reviews': Amazon.get_number_of_reviews(**book)})
                info_list.append(info)
            except Exception:
                continue
        return info_list

    @staticmethod
    def cursor_to_df(cursor):
        d = ms.dacc.mong.util.to_df(cursor, 'sale_ranks')
        d = process_sales_rank_category(d)
        return d

    @staticmethod
    def get_min_max_sales_rank_dates(book_info):
        cumul = list()
        for x in list(book_info['sales_rank'].values()):
            try:
                cumul += x['data']['date'].tolist()
            except Exception:
                raise
        return [np.min(cumul), np.max(cumul)]

    def mk_book_info(self, title, country, **kwargs):
        book_info = dict()
        kwargs = dict(kwargs, **self.default)
        d = AmazonBookWatch.cursor_to_df(self.db.find(spec={'title': title, 'country': country})
                        .sort([('_id', -1)]).limit(kwargs['max_date_ticks']))
        book_info['num_reviews'] = np.max(d['num_of_reviews'])
        book_info['sales_rank'] = dict()
        d = d[['date', 'sales_rank_category', 'sales_rank_subcategory', 'sales_rank']]
        categories = np.unique(d['sales_rank_category'])
        for c in categories:
            dd = d[d['sales_rank_category'] == c].sort('date', ascending=True)
            book_info['sales_rank'][c] = dict()
            book_info['sales_rank'][c]['sales_rank_subcategory'] = dd['sales_rank_subcategory'].iloc[0]
            dd = dd[['date', 'sales_rank']]
            book_info['sales_rank'][c]['data'] = dd
            ddd = dd[dd['date'] > datetime.now() - timedelta(days=kwargs['stats_num_of_days'])]
            book_info['sales_rank'][c]['rank_stats'] = pd.DataFrame([{
                'hi_rank': np.min(ddd['sales_rank']),
                'mean_rank': np.round(np.mean(ddd['sales_rank'])),
                'lo_rank': np.max(ddd['sales_rank'])
            }])
            book_info['sales_rank'][c]['rank_stats'] = \
                book_info['sales_rank'][c]['rank_stats'][['hi_rank', 'mean_rank', 'lo_rank']]
        book_info['commun_date_range'] = self.get_min_max_sales_rank_dates(book_info)
        return book_info

    def mk_sales_rank_plot(self, d, category='', save_filename=True, **kwargs):
        kwargs = dict(kwargs, **self.default)
        if isinstance(d, dict):
            if 'sales_rank' in list(d.keys()):
                d = d['sales_rank'][category]['data']
            elif category in list(d.keys()):
                d = d[category]['data']
            elif 'data' in list(d.keys()):
                d = d['data']
            else:
                raise ValueError('Your dict must have a "data" key or a %s key' % category)
        d = d.sort('date')
        x = [xx.to_datetime() for xx in d['date']]
        y = list(d['sales_rank'])

        gap_thresh = timedelta(seconds=kwargs['frequency_in_hours'] * 4.1 * 3600)
        x, y = insert_nans_in_x_and_y_when_there_is_a_gap_in_x(x, y, gap_thresh=gap_thresh)
        fig, ax = plt.subplots(1)

        fig.set_figheight(kwargs['figheight'])
        fig.set_figwidth(kwargs['figwidth'])
        ax.plot(x, y, kwargs['line_style'], linewidth=kwargs['linewidth'])
        commun_date_range = kwargs.get('commun_date_range', None)
        if commun_date_range:
            pylab.xlim(kwargs['commun_date_range'])
        ax.fill_between(x, y, max(y), facecolor=kwargs['facecolor'], alpha=0.5)

        # plt.ylabel('Amazon (%s) Sales Rank' % category, fontsize=kwargs['label_fontsize'])
        plot_title = kwargs.get('plot_title', 'Amazon (%s) Sales Rank' % category)
        plt.title(plot_title, fontsize=kwargs['title_fontsize'])

        plt.tick_params(axis='y', which='major', labelsize=kwargs['tick_font_size'])
        # plt.tick_params(axis='x', which='major', labelsize=kwargs['tick_font_size'])
        plt.tick_params(axis='x', which='minor', labelsize=kwargs['tick_font_size'])

        plt.gca().invert_yaxis()
        # ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1), interval=1))
        ax.xaxis.set_minor_locator(dates.DayLocator(interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%a\n%d %b'))
        ax.xaxis.grid(True, which="minor")
        ax.yaxis.grid()
        ax.xaxis.set_major_locator(dates.MonthLocator())
        # ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n%b\n%Y'))
        plt.tight_layout()
        if save_filename:
            if isinstance(save_filename, str):
                save_filename = save_filename + '.' + kwargs['save_format']
            else:  # save to temp file
                save_filename = tempfile.NamedTemporaryFile().name
            plt.savefig(save_filename, format=kwargs['save_format'], dpi=kwargs['dpi'])
            return save_filename
        else:
            return None

    def mk_book_info_html(self, title, country, **kwargs):
        kwargs = dict(kwargs, **self.default)
        book_info = self.mk_book_info(title, country, **kwargs)

        html = kwargs['book_info_html_template'].format(
            book_title=title,
            country=country,
            num_of_reviews=book_info['num_reviews']
        )
        html = html + "<br>\n"
        for category in list(book_info['sales_rank'].keys()):
            # make and save a graph, send to s3, and return a url for it
            file_name = self.mk_sales_rank_plot(
                d=book_info['sales_rank'],
                category=category, save_filename=True,
                commun_date_range=book_info['commun_date_range'],
                plot_title='Amazon.%s (%s) Sales Rank' % (
                    country, book_info['sales_rank'][category]['sales_rank_subcategory']),
                **kwargs
            )
            s3_key_name = '{title} - {country} - {category} - {date}.png'.format(
                title=title,
                country=country,
                category=category,
                date=datetime.now().strftime('%Y%m%d')
            )
            self.s3.dumpf(file_name, s3_key_name)
            image_url = self.s3.get_http_for_key(s3_key_name)
            html = html + kwargs['category_html'].format(
                image_url=image_url
            ) + "<br>\n"
        # html = html + "\n<br>"
        return html

    def mk_html_report(self, title_country_list=None):
        title_country_list = title_country_list or [
            {'title': 'Never mind the bullocks', 'country': 'co.uk'},
            {'title': 'Never mind the bullocks', 'country': 'com'},
            {'title': 'The Nanologues', 'country': 'in'}
        ]
        html = ''

        html += 'Stats of the last 24 hours:<br>'
        d = pd.DataFrame()
        for title_country in title_country_list:
            title = title_country['title']
            country = title_country['country']
            book_info = self.mk_book_info(title=title, country=country)
            for category in list(book_info['sales_rank'].keys()):
                dd = pd.concat([pd.DataFrame([{'title': title, 'country': country, 'category': category}]),
                                book_info['sales_rank'][category]['rank_stats']], axis=1)
                d = pd.concat([d, dd])
        d = d[['title', 'country', 'category', 'lo_rank', 'mean_rank', 'hi_rank']]

        html += ms.daf.to.to_html(d, template='box-table-c', index=False, float_format=lambda x: "{:,.0f}".format(x))

        # html += d.to_html(index=False).replace(
        #                        '<table border="0" class="dataframe">\n  ',
        #                        '<table border="0" class="dataframe" border-collapse:collapse >\n  ')
        # html += '<br>\n'

        # for title_country in title_country_list:
        #     title = title_country['title']
        #     country = title_country['country']
        #     html += '<p> {title} (in {country}) average ranks: '.format(title=title, country=country)
        #     book_info = self.mk_book_info(title=title, country=country)
        #     for category in book_info['sales_rank'].keys():
        #         html += '{category}: {rank}   '.format(category=category,
        #                                                rank=int(book_info['sales_rank'][category]
        #                                                ['rank_stats']['mean_rank'].iloc[0]))
        for title_country in title_country_list:
            title = title_country['title']
            country = title_country['country']
            html += self.mk_book_info_html(title=title, country=country)

        return html


# def last_of_list_if_list(d):
#     if isinstance(d['sales_rank_category'], list):
#         d['sales_rank_category'] = d['sales_rank_category'][-1]

def process_sales_rank_category(d):
    d['sales_rank_subcategory'] = [' > '.join(x) if isinstance(x, list) else x for x in d['sales_rank_category']]
    d['sales_rank_category'] = [x[-1] if isinstance(x, list) else x for x in d['sales_rank_category']]
    return d