__author__ = 'thorwhalen'

import numpy as np
import ut.daf.ch as daf_ch
import ut.semantics.math as semantics_math

import nltk
import pandas as pd


class TermWeightGetter(object):
    def __init__(self, term_gweight_selector):
        self.term_gweight_selector = term_gweight_selector

    def termcount_to_termweight(self, termstats):
        termstats = termcount_to_termtf(termstats)
        term_gweight = self.term_gweight_selector.get_table(list(termstats.index))
        return times(termstats, term_gweight)


def times(sr1, sr2):
    return (sr1*sr2).dropna()


def cosine(sr1, sr2):
    return (
        (sr1*sr2).dropna().sum()
            / (np.linalg.norm(sr1.values)*np.linalg.norm(sr2.values))
    )[0]


def termcount_to_termtf(termcount):
    """
    returns a term-frequency series given a term-count series
    """
    df = termcount / float(termcount.sum())
    df.columns = ['stat']
    return df


def termdoc_to_termdoc_count(term_doc_df, doc_var=None, term_var='term', count_var='count'):
    # processing input
    term_doc_df, doc_var, term_var = __process_term_doc_var_names__(term_doc_df, doc_var=doc_var, term_var=term_var)
    term_doc_df = term_doc_df.groupby([doc_var, term_var]).count()
    term_doc_df = daf_ch.ch_col_names(term_doc_df, count_var, term_var)
    return term_doc_df


def termdoc_to_doc_counts(term_doc_df, doc_var=None, term_var='term', count_var='count'):
    term_doc_df, doc_var, term_var = __process_term_doc_var_names__(term_doc_df, doc_var=doc_var, term_var=term_var)
    # keep only doc and terms, and one copy of any (doc,term) pair
    term_doc_df = term_doc_df[[doc_var, term_var]].drop_duplicates(cols=[doc_var, term_var]).reset_index(drop=True)
    # group by terms
    term_doc_df = term_doc_df[[term_var]].groupby(term_var).count()
    term_doc_df = daf_ch.ch_col_names(term_doc_df, count_var, term_var)
    return term_doc_df


def termdoc_to_term_idf(term_doc_df, doc_var=None, term_var='term'):
    # processing input
    term_doc_df, doc_var, term_var = __process_term_doc_var_names__(term_doc_df, doc_var=doc_var, term_var=term_var)
    # get the number of docs
    num_of_docs = len(np.unique(term_doc_df[doc_var]))
    # get doc_counts
    term_doc_df = termdoc_to_doc_counts(term_doc_df, doc_var, term_var)
    # # keep only doc and terms, and one copy of any (doc,term) pair
    # term_doc_df = term_doc_df[[doc_var,'term']].drop_duplicates(cols=[doc_var, 'term']).reset_index(drop=True)
    # # group by terms
    # term_doc_df = term_doc_df[['term']].groupby('term').count()
    term_doc_df['term'] = \
        semantics_math.idf_log10(num_of_docs_containing_term=np.array(term_doc_df['term']), num_of_docs=float(num_of_docs))
    return daf_ch.ch_col_names(term_doc_df, ['stat'], [term_var])


def __process_term_doc_var_names__(term_doc_df, doc_var=None, term_var='term'):
    if term_var != 'term':
        term_doc_df = daf_ch.ch_col_names(term_doc_df, [term_var], [term_var])
    cols = term_doc_df.columns
    if doc_var is None:  # try to guess it
        if len(cols) != 2:
            raise ValueError("In order to guess the doc_var, there needs to be only two columns")
        else:
            doc_var = list(set(cols)-set([term_var]))[0]
    return (term_doc_df, doc_var, term_var)


def mk_termCounts(dat, indexColName, strColName, data_folder=''):
    """
    input: data_folder='', dataname, savename='',
    output: string of ascii char correspondents
      (replacing, for example, accentuated letters with non-accentuated versions of the latter)
    """
    from ut.util import log
    from ut.daf.get import get_data
    dat = get_data(dat,data_folder)
    log.printProgress("making {} word counts (wc)",strColName)
    sr = to_kw_fd(dat[strColName])
    sr.index = dat.hotel_id.tolist()
    return sr
    # translate fds to series
    #printProgress("translating {} FreqDist to Series",col)
    #sr = fd_to_series(sr)
    #printProgress("saving {} word counts (wc)",col)
    #save_data(sr,savename + col)
    #printProgress('Done!')


def to_kw_tokens(dat, indexColName, strColName, data_folder=''):
    """
    input: daf of strings
    output: series of the tokens of the strings, processed for AdWords keywords
      i.e. string is lower capsed and asciied, and words are [\w&]+
    """
    from ut.util import log
    import ut.pstr.trans
    from ut.daf.get import get_data
    dat = get_data(dat, data_folder)
    log.printProgress("making {} tokens",strColName)
    sr = dat[strColName]
    sr.index = dat.hotel_id.tolist()
    # preprocess string
    sr = sr.map(lambda x: x.lower())
    sr = sr.map(lambda x: ut.pstr.trans.toascii(x))
    # tokenize
    sr = sr.map(lambda x:nltk.regexp_tokenize(x,'[\w&]+'))
    # return this
    return sr


def to_kw_fd(s):
    """
    input: string, series (with string columns)
    output: nltk.FreqDist (word count) of the tokens of the string, processed for AdWords keywords
      i.e. string is lower capsed and asciied, and words are [\w&]+
    """
    import ut.pstr.trans
    if isinstance(s,pd.Series):
        # preprocess string
        s = s.map(lambda x: x.lower())
        s = s.map(lambda x: ut.pstr.trans.toascii(x))
        # tokenize
        s = s.map(lambda x:nltk.regexp_tokenize(x,'[\w&]+'))
        # return series of fds
        return s.map(lambda tokens:nltk.FreqDist(tokens))
    elif isinstance(s,str):
        # preprocess string
        s = ut.pstr.trans.toascii(s.lower())
        # tokenize
        tokens = nltk.regexp_tokenize(s,'[\w&]+')
        return nltk.FreqDist(tokens)


def fd_to_series(fd):
    """
    input: nltk.FreqDist, pd.Series (with FreqDist in them), or list of FreqDists
    output: the pd.Series representation of the FreqDist
    """
    if isinstance(fd,nltk.FreqDist):
        return pd.Series(fd.values(),fd.keys())
    elif isinstance(fd,pd.Series):
        return fd.map(lambda x:pd.Series(x.values(),x.keys()))
    elif isinstance(fd,list):
        return map(lambda x:pd.Series(x.values(),x.keys()))