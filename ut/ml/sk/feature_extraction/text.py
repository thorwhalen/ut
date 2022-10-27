__author__ = 'thor'

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from pandas import Series, DataFrame
from numpy import array, int64, empty, hstack
from inspect import getargvalues, currentframe
import re
from collections import Counter
from itertools import chain
from urllib.parse import urlsplit

path_separator_pattern = re.compile('/+')
word_inclusion_pattern = re.compile('\w+')
path_inclusion_pattern = re.compile('[^/]+')
url_extension_pattern = re.compile('\.\w+$')


class TreeTokenizer(BaseEstimator, TransformerMixin):
    """
    A tokenizer that is the result of multiple different tokenizers that might either all be applied to the
    same text, or are used recursively to break up the text into finner pieces.
    In a deep_tokenizer tokenizers[0] tokenizes the text, then the next tokenizer, tokeizers[1] is applied to these
    tokens, and so forth. By default, the union of the tokens are returned. If token_prefixes is specified (usually,
    a different one for each tokenizer), they are prepended to the tokens to distinguish what level of tokenization
    they come from.

    If text_collection is specified, along with max_df and/or min_df, the text_collection will serve to learn a
    vocabulary for each tokenizer by collecting only those tokens whose frequency is at least min_df and no more than
    max_df.

    Params:
        * tokenizers: A list of tokenizers (function taking text and outputing a list of strings
        * token_prefixes: A list of prefixes to add in front of the tokens matched for each tokenizer
        (or a single string that will be used for all tokenizers
        * max_df and min_df: Only relevant when leaning text_collection is specified.
        These are respectively the max and min frequency that tokens should have to be included.
        The frequency can be expressed as a count, or a ratio of the total count.
        Note that in the case of max_df, it will always be relative to the total count of tokens at the current level.
        * return_tokenizer_info: Boolean (default False) indicating whether to return the tokenizer_info_list as well

    Fit input X is a collection of the text to learn the vocabulary with

    >>> from ut.ml.feature_extraction.text import mk_deep_tokenizer
    >>> import re
    >>>
    >>> t = [re.compile('[\w-]+').findall, re.compile('\w+').findall]
    >>> p = ['level_1=', 'level_2=']
    >>> ttok = TreeTokenizer(tokenizers=t, token_prefixes=p)
    >>> ttok.tokenize('A-B C B')
    ['level_1=A-B', 'level_1=C', 'level_1=B', 'level_2=A', 'level_2=B', 'level_2=C', 'level_2=B']
    >>> s = ['A-B-C A-B A B', 'A-B C B']
    >>> ttok = TreeTokenizer(tokenizers=t, token_prefixes=p, min_df=2)
    >>> _ = ttok.fit(text_collection=s)
    >>> ttok.transform(['A-B C B']).tolist()
    [['level_1=B', 'level_1=A-B', 'level_2=C']]
    """

    def __init__(
        self,
        preprocessor=None,
        tokenizers=[lambda x: [x], word_inclusion_pattern.findall],
        token_prefixes='',
        max_df=1.0,
        min_df=1,
        max_features=None,
        stop_words=[],
        output_of_tokenizer_with_input_not_a_string=[],
        keep_tokens_count=False,
    ):
        args, _, _, values = getargvalues(currentframe())
        values.pop('self')
        for arg, val in list(values.items()):
            setattr(self, arg, val)

        self._fitted = False

        if max_features is not None:
            raise NotImplementedError("max_features isn't implemented yet")

    def fit(self, text_collection, y=None):
        n_tokenizers = len(self.tokenizers)
        if not isinstance(self.token_prefixes, str):
            assert n_tokenizers == len(self.token_prefixes), (
                'Either all tokenizers must have the same prefix, '
                'or you should specify as many prefixes as there are tokenizers'
            )
        else:
            self.token_prefixes = [self.token_prefixes] * n_tokenizers

        if self.preprocessor is not None:
            text_collection = list(
                map(
                    self.preprocessor,
                    [x for x in text_collection if isinstance(x, str)],
                )
            )
        else:
            text_collection = [x for x in text_collection if isinstance(x, str)]

        if text_collection is not None:
            n = len(text_collection)
            if self.max_df > 1:
                self.max_df /= n
            if self.min_df < 1:
                self.min_df *= n

            # make the needed data structures
            self.tokenizer_info_list_ = list()
            for i in range(n_tokenizers):
                this_tokenizer_info = dict()
                this_tokenizer_info['tokenize'] = self.tokenizers[i]
                this_tokenizer_info['token_prefix'] = self.token_prefixes[i]
                this_tokenizer_info['vocab'] = set([])
                self.tokenizer_info_list_.append(this_tokenizer_info)

            # initialize remaining_element_counts to everything (with counts set to 1)
            remaining_element_counts = Counter(text_collection)
            max_df_thresh = self.max_df * len(text_collection)

            for i, tokenizer_info in enumerate(self.tokenizer_info_list_):
                # filter(url_word_tokens_count.update,
                # chain(*imap(lambda kv: [{word: kv[1]} for word in word_separation_pattern.findall(kv[0])],
                # remaining_element_counts.iteritems())))

                # initialize tokens_count
                tokens_count = Counter()
                # accumulate the counts of the tokens created by the current tokenizer
                list(
                    filter(
                        tokens_count.update,
                        chain(
                            *map(
                                lambda kv: [
                                    {token: kv[1]}
                                    for token in tokenizer_info['tokenize'](kv[0])
                                ],
                                iter(remaining_element_counts.items()),
                            )
                        ),
                    )
                )
                if len(tokens_count) > 0:  # if we got anything...
                    # ... remember the vocabulary
                    tokens_count = Series(tokens_count)
                    # get rid of what's too frequent
                    tokens_count = tokens_count[tokens_count <= max_df_thresh]
                    # add anything frequent enough in this tokenizer's vocabulary
                    min_lidx = tokens_count >= self.min_df
                    vocab_tokens_count = tokens_count[min_lidx]
                    tokenizer_info['vocab'] = set(
                        vocab_tokens_count.index.values
                    ).difference(self.stop_words)
                    if self.keep_tokens_count or self.max_features is not None:
                        self.tokenizer_info_list_[i][
                            'tokens_count'
                        ] = vocab_tokens_count.to_dict()
                    # what's not frequent enough will be treated by the next tokenizer
                    remaining_element_counts = tokens_count[~min_lidx].to_dict()
                    max_df_thresh = self.max_df * len(remaining_element_counts)
                else:  # no need to go further
                    break

            if self.max_features is not None:
                pass

            self._fitted = True

        return self

    def tokenize(self, text):
        if isinstance(text, str):
            if self.preprocessor is not None:
                text = self.preprocessor(text)
            if not self._fitted:
                tokens = []
                to_be_tokenized_further = [text]
                for level_tokenizer, token_prefix in zip(
                    self.tokenizers, self.token_prefixes
                ):
                    to_be_tokenized_further = list(
                        chain(*map(level_tokenizer, to_be_tokenized_further))
                    )
                    if (
                        len(to_be_tokenized_further) > 0
                    ):  # if any tokens were matched...
                        # ... keep them
                        tokens.extend(
                            [token_prefix + x for x in to_be_tokenized_further]
                        )
                    else:  # if not, we're done
                        break
                return tokens
            else:
                tokens = []
                to_be_tokenized_further = [text]
                for tokenizer_info in self.tokenizer_info_list_:
                    if len(to_be_tokenized_further) > 0:
                        to_be_tokenized_further = set(
                            chain(
                                *map(
                                    tokenizer_info['tokenize'], to_be_tokenized_further
                                )
                            )
                        )
                        # to_be_tokenized_further = set(map(tokenizer_info['tokenize'], to_be_tokenized_further))
                        matched_tokens = to_be_tokenized_further.intersection(
                            tokenizer_info['vocab']
                        )
                        if len(matched_tokens) > 0:  # if any tokens were matched...
                            # ... keep them
                            tokens.extend(
                                [
                                    tokenizer_info['token_prefix'] + x
                                    for x in matched_tokens
                                ]
                            )
                            # and don't tokenize them further
                            to_be_tokenized_further = to_be_tokenized_further.difference(
                                matched_tokens
                            )
                    else:
                        break
                return tokens
        else:
            return self.output_of_tokenizer_with_input_not_a_string

    def transform(self, text_collection):
        return array(list(map(self.tokenize, text_collection)))

    def token_list(self):
        token_list_ = []
        for info in self.tokenizer_info_list_:
            token_list_.extend([info['token_prefix'] + x for x in info['vocab']])
        return token_list_

    @classmethod
    def mk_url_tree_tokenizer(
        cls,
        urls=None,
        preprocessor=lambda x: url_extension_pattern.sub('', x.lower()),
        tokenizers=[
            lambda url: [urlsplit(url.lower()).path],
            path_inclusion_pattern.findall,
            word_inclusion_pattern.findall,
        ],
        token_prefixes=['url=', 'url_section=', 'url_word='],
        max_df=1.0,
        min_df=1,
        max_features=None,
        stop_words=['aspx', 'html', 'asp', 'cgi'],
        keep_tokens_count=False,
    ):
        tree_tokenizer = TreeTokenizer(
            preprocessor=preprocessor,
            tokenizers=tokenizers,
            token_prefixes=token_prefixes,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            stop_words=stop_words,
            keep_tokens_count=keep_tokens_count,
        )
        if urls is None:
            return tree_tokenizer
        else:
            return tree_tokenizer.fit(urls)


class MultiTreeTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, tree_tokenizers=None):
        self.tree_tokenizers = tree_tokenizers

    def fit(self, X, y=None):
        if self.tree_tokenizers is None:
            self.tree_tokenizers = [TreeTokenizer()]
        for i, tree_tokenizer in enumerate(self.tree_tokenizers):
            self.tree_tokenizers[i].fit(X[:, i])
        return self

    def tokenize(self, single_row):
        return hstack(
            tuple(
                [
                    self.tree_tokenizers[i].tokenize(single_row[i])
                    for i in range(len(self.tree_tokenizers))
                ]
            )
        )

    def transform(self, X):
        # return map(self.tokenize, X)
        separate_tokens = map(
            lambda i: self.tree_tokenizers[i].transform(X[:, i]),
            range(len(self.tree_tokenizers)),
        )
        return [
            hstack(tuple(x))
            for x in array([x.reshape(len(X)) for x in separate_tokens]).T
        ]
        # return map(self.tokenize, X)

    def token_list(self):
        token_list_ = []
        for tree_tokenizer in self.tree_tokenizers:
            token_list_.extend(tree_tokenizer.token_list())
        return token_list_


class TreeCountVectorizer(CountVectorizer):
    def __init__(
        self,
        preprocessor=None,
        tokenizers=[lambda x: [x], word_inclusion_pattern.findall],
        token_prefixes='',
        max_df=1.0,
        min_df=1,
        max_features=None,
        output_of_tokenizer_with_input_not_a_string=[],
        input='content',
        encoding='utf-8',
        decode_error='strict',
        strip_accents=None,
        lowercase=False,
        stop_words=None,
        ngram_range=(1, 1),
        analyzer='word',
        binary=False,
        dtype=int64,
        tokenizer=None,
        vocabulary={},
    ):
        args, _, _, values = getargvalues(currentframe())
        values.pop('self')
        values.pop('preprocessor')
        # kwargs = {}
        for arg, val in list(values.items()):
            setattr(self, arg, val)

        if preprocessor is None:
            self.preprocessor = lambda x: x if isinstance(x, str) else ''
        else:
            self.preprocessor = lambda x: preprocessor(x) if isinstance(x, str) else ''

        self.tree_tokenizer = TreeTokenizer(
            preprocessor=self.preprocessor,
            tokenizers=self.tokenizers,
            token_prefixes=self.token_prefixes,
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            output_of_tokenizer_with_input_not_a_string=self.output_of_tokenizer_with_input_not_a_string,
        )
        if not self.vocabulary:
            self.vocabulary = {}

    def fit(self, raw_documents=None, y=None):
        self.tree_tokenizer.fit(text_collection=raw_documents)
        self.set_params(
            tokenizer=self.tree_tokenizer.tokenize,
            vocabulary={w: i for i, w in enumerate(self.token_list())},
        )
        super(TreeCountVectorizer, self).fit_transform(raw_documents=raw_documents)
        return self

    def transform(self, raw_documents):
        return super(TreeCountVectorizer, self).transform(raw_documents=raw_documents)

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents).transform(raw_documents)

    def tokens_of(self, raw_documents):
        return self.inverse_transform(self.transform(raw_documents=raw_documents))


class MultiTreeCountVectorizer(CountVectorizer):
    def __init__(
        self,
        tree_tokenizers=None,
        input='content',
        preprocessor=None,
        encoding='utf-8',
        decode_error='strict',
        strip_accents=None,
        lowercase=False,
        max_df=1.0,
        min_df=1,
        max_features=None,
        stop_words=None,
        ngram_range=(1, 1),
        analyzer='word',
        binary=False,
        dtype=int64,
        tokenizer=None,
        vocabulary={},
    ):
        args, _, _, values = getargvalues(currentframe())
        values.pop('self')
        # kwargs = {}
        for arg, val in list(values.items()):
            setattr(self, arg, val)

    def fit(self, X, y=None):
        self.multi_tree_tokenizer_ = MultiTreeTokenizer(self.tree_tokenizers).fit(X)
        self.set_params(
            tokenizer=self.multi_tree_tokenizer_.tokenize,
            vocabulary={
                w: i for i, w in enumerate(self.multi_tree_tokenizer_.token_list())
            },
        )
        super(MultiTreeCountVectorizer, self).fit_transform(raw_documents=X)
        return self

    def transform(self, X):
        return super(MultiTreeCountVectorizer, self).transform(raw_documents=X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
