"""Text feature extraction"""

__author__ = 'thor'

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from pandas import Series
import numpy as np
from urllib.parse import urlsplit
import re
from collections import Counter
from itertools import chain
from operator import add

from ut.ml.sk.feature_extraction.text import TreeTokenizer
from functools import reduce

path_separator_pattern = re.compile('/+')
word_inclusion_pattern = re.compile("\w+")
path_inclusion_pattern = re.compile("[^/]+")


def mk_url_count_vectorizer(preprocessor=lambda url: urlsplit(url.lower()).path,
                            max_df=1.0, min_df=1, max_features=None,
                            binary=True, **kwargs):
    tokenizer = TreeTokenizer.mk_url_tree_tokenizer(max_df=max_df, min_df=min_df).tokenize

    return CountVectorizer(preprocessor=preprocessor, tokenizer=tokenizer,
                           max_df=max_df, min_df=min_df, max_features=max_features, binary=binary, **kwargs)


class MultiVarTextVectorizer(BaseEstimator):
    """
    MultiVarTextVectorizer is the "multi-dimensional corpus" version of CountVectorizer.
    You use this when instead of items being text documents, items are a collection of documents of different types.

    A MultiVarTextVectorizer is vectorizes as CountVectorizer does, but:
        * where CountVectorizer fits from a corpus of docs, MultiVarTextVectorizer fits from a dataframe with some
        specified text columns (or array of dicts with specified text fields).
        * where CountVectorizer transforms text to a vector, MultiVarTextVectorizer transforms Series, or dicts, into
        vectors, applying a different tokenizer to each different field/variable of the input item,
        returning the union of the resulting tokens as the set to be vectorized.

    Input:
        * var_tokenizers: a dict of {var_name: Tokenizer} pairs that specifies how to tokenize each
    """
    def __init__(self, var_tokenizers, count_vectorizer_kwargs):
        self.set_params(var_tokenizers=var_tokenizers)

    def set_params(self, **parameters):
        for parameter, value in list(parameters.items()):
            self.setattr(parameter, value)
        self.tokenized_cols_ = list(self.var_tokenizers.keys())
        return self

    def get_params(self, deep=True):
        return {
            'var_tokenizers': self.var_tokenizers
        }

    def fit(self, X, y=None):
        if not set(self.tokenized_cols_).issubset(X.columns):
            raise ValueError("All keys of var_tokenizers must be present as columns of X")

        X = X[self.tokenized_cols_]
        self.tokenizer_ = \
            lambda d: reduce(add, [tokenize(d[var]) for var, tokenize in self.var_tokenizers.items()], list())

        self.count_vectorizer = CountVectorizer(tokenizer=self.tokenizer_, )



# class DeepTokenizer()


def mk_deep_tokenizer(text_collection=None,
                      tokenizers=[lambda x: [x], word_inclusion_pattern.findall],
                      token_prefixes='',
                      max_df=1.0,
                      min_df=1,
                      return_tokenizer_info=False):
    """
    Makes a tokenizer that is the result of multiple different tokenizers that might either all be applied to the
    same text, or are used recursively to break up the text into finner pieces.
    In a deep_tokenizer tokenizers[0] tokenizes the text, then the next tokenizer, tokeizers[1] is applied to these
    tokens, and so forth. By default, the union of the tokens are returned. If token_prefixes is specified (usually,
    a different one for each tokenizer), they are prepended to the tokens to distinguish what level of tokenization
    they come from.

    If text_collection is specified, along with max_df and/or min_df, the text_collection will serve to learn a
    vocabulary for each tokenizer by collecting only those tokens whose frequency is at least min_df and no more than
    max_df.

    Input:
        * text_collection: a collection of the text to learn the vocabulary with
        * tokenizers: A list of tokenizers (function taking text and outputing a list of strings
        * token_prefixes: A list of prefixes to add in front of the tokens matched for each tokenizer
        (or a single string that will be used for all tokenizers
        * max_df and min_df: Only relevant when leaning text_collection is specified.
        These are respectively the max and min frequency that tokens should have to be included.
        The frequency can be expressed as a count, or a ratio of the total count.
        Note that in the case of max_df, it will always be relative to the total count of tokens at the current level.
        * return_tokenizer_info: Boolean (default False) indicating whether to return the tokenizer_info_list as well


    >>> from ut.ml.feature_extraction.text import mk_deep_tokenizer
    >>> import re
    >>>
    >>> t = [re.compile('[\w-]+').findall, re.compile('\w+').findall]
    >>> p = ['level_1=', 'level_2=']
    >>> tokenizer = mk_deep_tokenizer(tokenizers=t, token_prefixes=p)
    >>>
    >>> tokenizer('A-B C B')
    ['level_1=A-B', 'level_1=C', 'level_1=B', 'level_2=A', 'level_2=B', 'level_2=C', 'level_2=B']
    >>> s = ['A-B-C A-B A B', 'A-B C B']
    >>> tokenizer = mk_deep_tokenizer(text_collection=s, tokenizers=t, token_prefixes=p, min_df=2)
    >>>
    >>> tokenizer('A-B C B')
    ['level_1=B', 'level_1=A-B', 'level_2=C']
    """
    raise DeprecationWarning("It's probably a better idea to use "
                             "ut.ml.sk.feature_extraction.text.TreeTokenizer().tokenizer")

    n_tokenizers = len(tokenizers)
    if not isinstance(token_prefixes, str):
        assert n_tokenizers == len(token_prefixes), \
            "Either all tokenizers must have the same prefix, " \
            "or you should specify as many prefixes as there are tokenizers"
    else:
        token_prefixes = [token_prefixes] * n_tokenizers

    if text_collection is None:
        # to_be_tokenized_further = ['free-will is slave-of will']
        # return list(chain(*imap(tokenizers[0], to_be_tokenized_further)))
        def tokenizer(text):
            tokens = []
            to_be_tokenized_further = [text]
            for level_tokenizer, token_prefix in zip(tokenizers, token_prefixes):
                to_be_tokenized_further = list(chain(*map(level_tokenizer, to_be_tokenized_further)))
                if len(to_be_tokenized_further) > 0:  # if any tokens were matched...
                    # ... keep them
                    tokens.extend([token_prefix + x for x in to_be_tokenized_further])
                else:  # if not, we're done
                    break
            return tokens

        return tokenizer
    else:

        n = len(text_collection)
        if max_df > 1:
            max_df /= n
        if min_df < 1:
            min_df *= n

        # make the needed data structures
        tokenizer_info_list = list()
        for i in range(n_tokenizers):
            this_tokenizer_info = dict()
            this_tokenizer_info['tokenize'] = tokenizers[i]
            this_tokenizer_info['token_prefix'] = token_prefixes[i]
            this_tokenizer_info['vocab'] = set([])
            tokenizer_info_list.append(this_tokenizer_info)

        # initialize remaining_element_counts to everything (with counts set to 1)
        remaining_element_counts = Counter(text_collection)
        max_df_thresh = max_df * len(text_collection)

        for i, tokenizer_info in enumerate(tokenizer_info_list):
            # filter(url_word_tokens_count.update,
            # chain(*imap(lambda kv: [{word: kv[1]} for word in word_separation_pattern.findall(kv[0])],
            # remaining_element_counts.iteritems())))

            # initialize tokens_count
            tokens_count = Counter()
            # accumulate the counts of the tokens created by the current tokenizer
            list(filter(tokens_count.update,
                   chain(*map(lambda kv: [{token: kv[1]} for token in tokenizer_info['tokenize'](kv[0])],
                               iter(remaining_element_counts.items())))))
            if len(tokens_count) > 0:  # if we got anything...
                # ... remember the vocabulary
                tokens_count = Series(tokens_count)
                # get rid of what's too frequent
                tokens_count = tokens_count[tokens_count <= max_df_thresh]
                # add anything frequent enough in this tokenizer's vocabulary
                min_lidx = tokens_count >= min_df
                tokenizer_info['vocab'] = set(tokens_count[min_lidx].index.values)
                # what's not frequent enough will be treated by the next tokenizer
                remaining_element_counts = tokens_count[~min_lidx].to_dict()
                max_df_thresh = max_df * len(remaining_element_counts)
            else:  # no need to go further
                break

        def tokenizer(text):
            tokens = []
            to_be_tokenized_further = [text]
            for tokenizer_info in tokenizer_info_list:
                if len(to_be_tokenized_further) > 0:
                    to_be_tokenized_further = \
                        set(chain(*map(tokenizer_info['tokenize'], to_be_tokenized_further)))
                    # to_be_tokenized_further = set(map(tokenizer_info['tokenize'], to_be_tokenized_further))
                    matched_tokens = to_be_tokenized_further.intersection(tokenizer_info['vocab'])
                    if len(matched_tokens) > 0:  # if any tokens were matched...
                        # ... keep them
                        tokens.extend([tokenizer_info['token_prefix'] + x for x in matched_tokens])
                        # and don't tokenize them further
                        to_be_tokenized_further = to_be_tokenized_further.difference(matched_tokens)
                else:
                    break
            return tokens

        if return_tokenizer_info:
            return tokenizer, tokenizer_info_list
        else:
            return tokenizer


# def mk_multiVar_text_vectorizer(var_tokenizers, count_vectorizer_kwargs):
#     tokenized_cols_ = var_tokenizers.keys()


def mk_url_tokenizer(urls=None, max_df=1.0, min_df=1, return_tokenizer_info=False):
    tokenizers = [lambda x: [x], path_inclusion_pattern.findall, word_inclusion_pattern.findall]
    token_prefixes = ['url=', 'url_section=', 'url_word=']
    return mk_deep_tokenizer(urls,
                             tokenizers=tokenizers,
                             token_prefixes=token_prefixes,
                             max_df=max_df,
                             min_df=min_df,
                             return_tokenizer_info=return_tokenizer_info)

