__author__ = 'thorwhalen'


import pandas as pd
import ut.pdict.ot as pdict_ot
import ut.util.ulist as util_ulist
from collections import OrderedDict
import re
import ut.pstr.trans as pstr_trans

# import ut.parse.html2text_formated as html2text_formated
from pattern.web import plaintext


# import venere.data_source as venere_data_source

########################################################################################################################
# FACTORIES
def lower_ascii_slash_w_terms():
    w_terms_re = re.compile(r'[^\w]+')
    w_terms = lambda x: ' '.join(w_terms_re.sub(' ', x).split()).strip()
    return TextProcessor(text_processors=[preprocess_text_lower_ascii, w_terms]).process


def lower_ascii_term_replacer(
    map_spec, rep_col=None, by_col=None, term_padding_exp=r'\b'
):
    term_replacer = TermReplacer(
        map_spec, rep_col=rep_col, by_col=by_col, term_padding_exp=term_padding_exp
    )
    return TextProcessor(
        text_processors=[preprocess_text_lower_ascii, term_replacer.replace_terms]
    ).process


def erenev_kw_str_term_replacer(rep_col=None, by_col=None, term_padding_exp=r'\b'):
    DeprecationWarning(
        'ut.semantics.text_processors.erenev_kw_str_term_replacer is depreciated: '
        'Use the ut.erenev.aw version instead'
    )
    print(
        'misc.semantics.text_processors.erenev_kw_str_term_replacer is depreciated: '
        'Use the ut.venere.aw version instead'
    )
    # venere_term_replacer = TermReplacer(venere_data_source.term_map, rep_col=None, by_col=None, term_padding_exp=r'\b')
    # return TextProcessor(text_processors=[aw_manip.kw_str, venere_term_replacer.replace_terms]).process


########################################################################################################################
# Class that composes mutilple text processors
class TextProcessor:
    def __init__(self, text_processors):
        self.text_processors = text_processors

    def process(self, text):
        for processor in self.text_processors:
            text = processor(text)
        return text

    # TODO: Replace TextProcessor composition using util_pfunc.multi_compose


########################################################################################################################
# A menu of text processors


def preprocess_text_lower_ascii(text):
    """
    Preprocesses the text before it will be fed to the tokenizer.
    Here, we should put things like lower-casing the text, casting letters to "simple" ("ascii", "non-accentuated")
    letters, replacing some common strings (such as "bed and breakfast", "New York" by singular token representatives
    such as "b&b", "new_york"), and what ever needs to be done before tokens are retrieved from text.
    """
    return pstr_trans.toascii(text).lower()


class TermReplacer:
    def __init__(self, map_spec, rep_col=None, by_col=None, term_padding_exp=r'\b'):
        if isinstance(map_spec, pd.DataFrame):
            # if map_spec is given by a dataframe, make a mapto dict out of it
            map_spec = pdict_ot.keyval_df(
                map_spec, key_col=rep_col, val_col=by_col, warn=True
            )

        self.pattern = list_to_token_matcher_re(
            list(map_spec.keys()), term_padding_exp=term_padding_exp
        )
        # replaces:
        # key_lengths = map(len,map_spec.keys())
        # keys = util_ulist.sort_as(map_spec.keys(), key_lengths, reverse=True)
        # self.pattern = re.compile(term_padding_exp + '(' + '|'.join(keys) + ')' + term_padding_exp)

        self.replace_by = lambda x: map_spec[x.group()]

    def replace_terms(self, s):
        return self.pattern.sub(self.replace_by, s)


def list_to_token_matcher_re(str_list, term_padding_exp=r'\b'):
    str_lengths = list(map(len, str_list))
    str_list = util_ulist.sort_as(str_list, str_lengths, reverse=True)
    return re.compile(
        term_padding_exp + '(' + '|'.join(str_list) + ')' + term_padding_exp
    )


def html2text(text):
    return plaintext(text)
    # return html2text_formated.html2text(text).replace('**', '')
