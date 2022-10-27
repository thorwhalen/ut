__author__ = 'thor'

import re
import pandas as pd
import numpy as np
from ut.semantics.text_processors import preprocess_text_lower_ascii
from ut.semantics.text_processors import html2text
from pattern.web import plaintext


class TermStats(object):
    default = dict()
    default['name'] = None
    default['term_col'] = 'term'
    default['stat_col'] = 'stat'
    default['html2text'] = plaintext
    default['text_preprocess'] = preprocess_text_lower_ascii
    default['tokenizer'] = re.compile('\w+').findall

    def __init__(self, data):
        self.sr = data

    def is_empty(self):
        return len(self.sr) == 0

    def len(self):
        return len(self.sr)

    def get_name(self):
        return self.sr.name

    def set_name(self, name):
        self.sr.name = name
        return self

    def terms(self):
        return self.sr.index.values

    def group_terms(self):
        name = self.sr.name
        self.sr = self.sr.groupby(level=0).sum()
        self.sr.name = name
        return self

    def total(self):
        return sum(self.sr.values)

    def normalize(self):
        self.sr = self.sr.groupby(level=0).sum()
        self.sr = self.sr / float(sum(self.sr.values))
        return self

    def normalized(self):
        return TermStats(
            self.sr.groupby(level=0).sum() / float(sum(self.sr.values))
        ).set_name(self.get_name())

    def dot(self, other):
        # return (self * other).total() # but chose to use the following for (supposed) efficiency
        t = np.nansum(self.sr * other.sr)
        if np.isnan(t):
            return 0
        else:
            return t

    def __mul__(self, other):
        return TermStats((self.sr * other.sr).dropna()).group_terms()

    def __div__(self, other):
        if isinstance(other, TermStats):
            return TermStats((self.sr / other.sr).dropna()).group_terms()
        else:
            return TermStats((self.sr / other).dropna()).group_terms()

    def __add__(self, other):
        return TermStats(pd.concat([self.sr, other.sr])).group_terms()

    def __str__(self):
        return self.sr.__str__()

    def __repr__(self):
        """
        This is used by iPython to display a variable.
        I choose to do thing differently than __str__ (eventually)
        """
        return self.sr.__repr__()

    def norm(self):
        return np.linalg.norm(self.sr.values)

    def to_dict(self):
        return {k: v for k, v in zip(self.sr.index, self.sr.values)}

    def sort(self):
        self.sr.sort(ascending=False)
        return self

    def head(self, *args, **kwargs):
        return self.sr.head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self.sr.tail(*args, **kwargs)

    @staticmethod
    def kernel(ts1, ts2):
        return (ts1 * ts2).total() / (ts1.norm() * ts2.norm())

    @staticmethod
    def cosine(ts1, ts2):
        return (ts1 * ts2).total() / (ts1.norm() * ts2.norm())

    @staticmethod
    def mk_empty():
        return TermStats(pd.Series())

    @staticmethod
    def from_df(df, **kwargs):
        """
        returns a TermStats by taking terms and stats from a "term_col" and "stat_col" of a dataframe,
         and grouping terms (by summing up all stats for a same term)
        """
        kwargs = dict(TermStats.default, **kwargs)
        return TermStats(
            pd.Series(data=df[kwargs['stat_col']], index=[kwargs['term_col']])
        ).group_terms()

    @staticmethod
    def from_dict(d, **kwargs):
        """
        returns a TermStats from a dict of term:stat pairs
        """
        kwargs = dict(TermStats.default, **kwargs)
        return TermStats(
            pd.Series(data=list(d.values()), index=list(d.keys()))
        ).set_name(kwargs['name'])

    @staticmethod
    def from_terms(terms, **kwargs):
        """
        returns a (count) TermStats from a list (or other iterable) of terms
        """
        kwargs = dict(TermStats.default, **kwargs)
        return (
            TermStats(pd.Series(data=1, index=terms))
            .group_terms()
            .set_name(kwargs['name'])
        )

    @staticmethod
    def from_text(text, **kwargs):
        """
        after preprocessing the text (default is lower ascii, but can be overwritten - by None for example),
        the function tokenizes the text (default is re.compile('\w+').findall) and returns a termcount
        """
        kwargs = dict(TermStats.default, **kwargs)
        if kwargs['text_preprocess']:
            text = kwargs['text_preprocess'](text)
        text = TermStats.default['tokenizer'](text)
        return TermStats.from_terms(text, **kwargs)

    @staticmethod
    def from_html(html, **kwargs):
        """
        transforms html to text with a given (or default) html2text method, and then calls from_text
        """
        kwargs = dict(TermStats.default, **kwargs)
        return TermStats.from_text(kwargs['html2text'](html), **kwargs)
