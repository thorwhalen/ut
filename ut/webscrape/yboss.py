__author__ = 'thor'

import re
import pandas as pd
import numpy as np
import pickle
import os

import ut as ms

# import ut.parse.html2text_formated as html2text_formated
from pattern.web import plaintext
import ut.pstr.trans
from ut.semantics.termstats import TermStats
from ut.slurp.yboss import Yboss


class YbossText(object):
    @staticmethod
    def flat_html(yb):
        return '\n'.join(yb['title']) + '\n'.join(yb['abstract'])

    @staticmethod
    def tc_flat(yb, **kwargs):
        return TermStats.from_html(YbossText.flat_html(yb), **kwargs)
        # text = YbossText.html2text('\n'.join(yb['title']) + '\n'.join(yb['abstract']))
        # text = text_preprocessor(text)
        # text = YbossText.simple_tokenizer(text)
        # tc = pd.DataFrame({'term': text})
        # tc['count'] = 1
        # return tc.groupby('term').sum().reset_index()

    @staticmethod
    def ascertain_df(yb):
        if isinstance(yb, pd.DataFrame):
            return yb
        else:
            if os.path.exists(yb):
                return YbossText.ascertain_df(pickle.load(open(yb, 'r')))
            else:
                raise NotImplementedError("Didn't implement this case yet")

    ########################################################################################
    ########### TEXT PROCESSING ############################################################

    @staticmethod
    def toascii_lower(text):
        return ms.pstr.trans.toascii(text).lower()

    @staticmethod
    def simple_tokenizer(text):
        text = re.compile('\W+').sub(' ', text).split(' ')
        if text[-1] == '':
            text = text[:-1]
        return text

    @staticmethod
    def remove_html_bold(s):
        return s.replace('<b>', '').replace('</b>', '')

    @staticmethod
    def html2text(text):
        return plaintext(text)
        # return html2text_formated.html2text(text).replace('**', '')


def compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


from ut.semantics.text_processors import TermReplacer
from oto.data_access.default_data_access_params import DefaultDataAccessParams


# from ut.semantics.termstats import TermStats


class YbossSemantics(object):
    def __init__(self, **kwargs):
        kwargs = dict(
            {
                'key_terms': list(),
                'term_map': {
                    'the': 'the'
                },  # dict() # TODO: dict() didn't work, so I put essentially empty {the:the}
                'yb_to_tc': self.tc_flat_from_yb,
                'yboss_kwargs': dict(),
            },
            **kwargs
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
        for w in self.key_terms:
            self.term_map = dict(self.term_map, **{w: '_' + w.replace(' ', '_')})
        self.key_terms_ts = TermStats.from_terms(list(self.term_map.values()))
        self.text_preprocess = compose(
            TermReplacer(self.term_map, term_padding_exp=r'\b').replace_terms,
            YbossText.toascii_lower,
        )
        self.yb = Yboss(**self.yboss_kwargs)
        delattr(self, 'yboss_kwargs')

    def tc_flat_from_yb(self, yb):
        return YbossText.tc_flat(yb, text_preprocess=self.text_preprocess)

    def tc_flat_from_html(self, html):
        return TermStats.from_html(html, text_preprocess=self.text_preprocess)

    def ss_key_term_kernel(self, tc):
        if isinstance(tc, str):
            tc = TermStats.from_html(tc, text_preprocess=self.text_preprocess)
        return self.key_terms_ts.dot(tc.normalize())

    def get_info_of_term(self, term):
        d = {'term': term}
        t = self.yb.slurp_content_as_dict(d['term'], service='limitedweb')[
            'bossresponse'
        ]['limitedweb']
        d['totalresults'] = t['totalresults']
        d['results'] = t['results']
        d['results_df'] = self.yb.content_to_results_df(t)
        d['flat_tc'] = self.tc_flat_from_yb(d['results_df']).sort()
        return d

    # def get_info_of_several_terms(self, term_list):
    #     d = dict()
    #     d['term_info'] = list()
    #     for i, term in enumerate(term_list):
    #         d['term_info'][i] = {'term': term}
    #         t = self.yb.slurp_content_as_dict(d['term1'], service='limitedweb')['bossresponse']['limitedweb']
    #         d['term_info']['totalresults'] = t['totalresults']
    #         d['term_info']['results'] = t['results']
    #         d['term_info']['results_df'] = self.yb.content_to_results_df(t)
    #         d['term_info']['']

    def cos_of_terms(self, term1, term2):
        if isinstance(term1, str):
            term1 = self.yb.slurp_content_as_dict(term1, service='limitedweb')[
                'bossresponse'
            ]['limitedweb']
        if isinstance(term1, dict):
            term1 = self.yb.content_to_results_df(term1)
        if not isinstance(term1, TermStats):
            term1 = self.tc_flat_from_yb(term1)
        if isinstance(term2, str):
            term2 = self.yb.slurp_content_as_dict(term2, service='limitedweb')[
                'bossresponse'
            ]['limitedweb']
        if isinstance(term2, dict):
            term2 = self.yb.content_to_results_df(term2)
        if not isinstance(term2, TermStats):
            term2 = self.tc_flat_from_yb(term2)
        return term1.normalize().dot(term2.normalize())

    def normalized_yahoo_distance(self, term1, term2):
        N = 2210000000  # took that to be the number of results for "the", but should be total number of pages indexed
        # http://arxiv.org/pdf/cs/0412098.pdf
        term1_total_results = np.log(
            int(
                self.yb.slurp_content_as_dict(term1, service='limitedweb')[
                    'bossresponse'
                ]['limitedweb']['totalresults']
            )
        )
        term2_total_results = np.log(
            int(
                self.yb.slurp_content_as_dict(term2, service='limitedweb')[
                    'bossresponse'
                ]['limitedweb']['totalresults']
            )
        )
        term1_term2_total_total_results = np.log(
            int(
                self.yb.slurp_content_as_dict(
                    term1 + ' ' + term2, service='limitedweb'
                )['bossresponse']['limitedweb']['totalresults']
            )
        )
        term2_term1_total_total_results = np.log(
            int(
                self.yb.slurp_content_as_dict(
                    term2 + ' ' + term1, service='limitedweb'
                )['bossresponse']['limitedweb']['totalresults']
            )
        )
        min_log = np.min([term1_total_results, term2_total_results])
        max_log = np.min([term1_total_results, term2_total_results])
        t12 = (max_log - term1_term2_total_total_results) / (np.log(N) - min_log)
        t21 = (max_log - term2_term1_total_total_results) / (np.log(N) - min_log)
        return np.mean([t12, t21])
