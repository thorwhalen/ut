__author__ = 'thorwhalen'

# import pandas as pd
import numpy as np
import ut.daf.ch as daf_ch
import ut.semantics.math as semantics_math

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