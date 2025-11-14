__author__ = 'thorwhalen'

import ut.parse.google as google
import pandas as pd
import re
from bs4.element import Tag
import ut.semantics.text_processors as semantics_text_processors
import ut.util.ulist as util_ulist
import ut.daf.manip as daf_manip
import ut.coll.order_conserving as colloc

#### Utilsxw

LOCATION_LOCAL = 'LOCAL'
LOCATION_S3 = 'S3'

split_exp_01 = re.compile(r'[^&\w]*')
tokenizer_re = re.compile(r'[&\w]+')


def mk_terms_df(df, text_cols, id_cols=None, tokenizer_re=tokenizer_re):
    text_cols = util_ulist.ascertain_list(text_cols)
    if id_cols is None:
        id_cols = colloc.setdiff(df.columns, text_cols)
    else:
        id_cols = util_ulist.ascertain_list(id_cols)
        id_cols_missing = colloc.setdiff(id_cols, df.columns)
        if (
            id_cols_missing
        ):  # if any columns are missing, try to get them from named index
            df = df.reset_index(id_cols_missing)
    dd = pd.DataFrame()
    for c in text_cols:
        d = df[id_cols]
        d['term'] = [re.findall(tokenizer_re, x) for x in df[c]]
        d = daf_manip.rollout_cols(d, cols_to_rollout='term')
        dd = pd.concat([dd, d])
    return dd


def space_seperated_token_string_to_term_count(s):
    return list_to_term_count(s.split(' '))


def list_to_term_count(term_list):
    """
    takes a token list and
    returns a Series whose indices are terms and values are term counts
    (the number of times the term appeared in the google result)
    """
    # make a dataframe of term counts TODO: Explore faster ways to do this
    df = pd.DataFrame(term_list, columns=['term'])
    df = df.groupby('term').count()
    df.columns = ['count']
    if len(df) == 1:
        df = pd.DataFrame({'term': term_list[:1], 'count': len(term_list)})
        df = df.set_index('term')
    return df


# def tokenize_text(gresult_text):
#     return re.split(split_exp,gresult_text)


def get_text_from_source_gresult(gresults):
    if not isinstance(
        gresults, dict
    ):  # if not a dict assume it's a soup, html, or filename thereof
        gresults = google.parse_tag_dict(google.mk_gresult_tag_dict(gresults))
    elif is_tag_dict(gresults):  # if gresults is a tag_dict, and make it a info dict
        gresults = google.parse_tag_dict(gresults)
    if 'organic_results_list' in gresults:
        title_text_concatinated = ' '.join(
            [
                x['title_text']
                for x in gresults['organic_results_list']
                if 'title_text' in x
            ]
        )
        snippet_text_concatinated = ' '.join(
            [x['st_text'] for x in gresults['organic_results_list'] if 'st_text' in x]
        )
        text_concatinated = title_text_concatinated + ' ' + snippet_text_concatinated
    else:
        search_for_tag = ['_ires', '_search', '_res', '_center_col']
        for t in search_for_tag:
            if t in gresults:
                text_concatinated = soup_to_text(gresults[t])
                break
        if not text_concatinated:  # if you still don't have anything
            text_concatinated = soup_to_text(
                gresults
            )  # ... just get the text from the whole soup
    return text_concatinated


def is_tag_dict(x):
    try:
        if isinstance(x[list(x.keys())[0]][0], Tag):
            return True
        else:
            return False
    except:
        return False


def soup_to_text(element):
    return list(filter(visible, element.findAll(text=True)))


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element), re.UNICODE):
        return False
    return True


class TokenizerFactor:
    @classmethod
    def simple(cls):
        split_exp = re.compile(r'[^&\w]*')

    def tokenizer(self, text):
        return re.split(self.split_exp, text)


class TokenizerFactory:
    @classmethod
    def get_simple_aw_tokenizer(cls):
        return cls.NegAlphabetTokenizer(split_exp=re.compile(r'[^&\w]*'))

    class NegAlphabetTokenizer:
        def __init__(self, split_exp):
            self.split_exp = split_exp

        def tokenize(self, text):
            return re.split(self.split_exp, text)


class TermStatsMaker:
    def __init__(
        self,
        get_text_from_source=get_text_from_source_gresult,
        preprocess_text=semantics_text_processors.preprocess_text_lower_ascii,
        tokenizer=TokenizerFactory.get_simple_aw_tokenizer().tokenize,
        mk_term_stats=list_to_term_count,
    ):
        self.get_text_from_source = get_text_from_source
        self.preprocess_text = preprocess_text
        self.tokenizer = tokenizer
        self.mk_term_stats = mk_term_stats

    def term_stats(self, text_source):
        # return self.mk_term_stats(self.tokenizer(self.preprocess_text(self.get_text_from_source(text_source))))
        text = self.get_text_from_source(text_source)
        precessed_text = self.preprocess_text(text)
        token_list = self.tokenizer(precessed_text)
        term_stats = self.mk_term_stats(token_list)
        return term_stats

        # # consider using composition as such:
        # return self.mk_term_stats(
        #     self.tokenize_text(
        #         self.preprocess_text(
        #             self.get_text_from_source(text_source)
        #         )
        #     )
        # )

    # @classmethod
    # def mk_term_stats_maker(cls):
    #     return TermStatsMaker(
    #         get_text_from_source=get_text_from_source_gresult,
    #         preprocess_text=semantics_text_processors.preprocess_text_lower_ascii,
    #         tokenizer=TokenizerFactory.get_simple_aw_tokenizer().tokenize,
    #         mk_term_stats=list_to_term_count
    #     )

    @classmethod
    def mk_term_stats_maker(cls):
        return TermStatsMaker(
            get_text_from_source=get_text_from_source_gresult,
            preprocess_text=semantics_text_processors.preprocess_text_lower_ascii,
            tokenizer=TokenizerFactory.get_simple_aw_tokenizer().tokenize,
            mk_term_stats=list_to_term_count,
        )

    @classmethod
    def mk_term_stats_maker_for_hotels(cls, term_map=None, location=LOCATION_LOCAL):
        print('oh no! you commented this out!!')
        # if term_map is None:
        #    import msvenere.factories as venere_factories
        #    if location==LOCATION_LOCAL:
        #        ds = venere_factories.data_source_for_local_term_stats_maker()
        #    elif location==LOCATION_S3:
        #        ds = venere_factories.data_source_for_s3_term_stats_maker()
        #    term_map = ds.d.term_map
        # return TermStatsMaker(
        #    get_text_from_source=get_text_from_source_gresult,
        #    preprocess_text=venere_aw.erenev_kw_str_term_replacer(),
        #    # preprocess_text=semantics_text_processors.lower_ascii_term_replacer(map_spec=term_map),
        #    tokenizer=TokenizerFactory.get_simple_aw_tokenizer().tokenize,
        #    mk_term_stats=list_to_term_count
        # )


# def mk_term_count_from_google_results(gresults):
#     """
#     takes a google result (in the form of html, filename thereof, soup, or info_dict, and
#     returns a Series whose indices are terms and values are term counts
#     (the number of times the term appeared in the google result)
#     """
#     # get preprocessed text from gresults
#     gresults = preprocess_text_lower_ascii(get_text_from_source_gresult(gresults))
#     # tokenize this text
#     toks = tokenize_text(gresults)
#     # make a dataframe of term counts TODO: Explore faster ways to do this
#     df = pd.DataFrame(toks,columns=['token'])
#     df = df.groupby('token').count()
#     df.columns = ['count']
#     df = df.sort(columns=['count'],ascending=False) # TODO: Take out sorting at some point since it's unecessary (just for diagnosis purposes)
#     return df
