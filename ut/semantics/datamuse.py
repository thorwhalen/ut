import unittest
import requests
from ut.pdict.to import Struct
import pandas as pd

__author__ = 'thor'

__doc__ = '''
python connector for http://www.datamuse.com/api/
Based on code from https://github.com/margaret/python-datamuse
'''

word_params_set = {
    'ml',
    'sl',
    'sp',
    'rel_jja',
    'rel_jjb',
    'rel_syn',
    'rel_ant',
    'rel_spc',
    'rel_gen',
    'rel_com',
    'rel_par',
    'rel_bga',
    'rel_bgb',
    'rel_rhy',
    'rel_nry',
    'rel_hom',
    'rel_cns',
    'v',
    'topics',
    'lc',
    'rc',
    'max',
}

word_params_struct = Struct({k: k for k in word_params_set})


class Datamuse:
    """
    See http://www.datamuse.com/api/

    Query parameters
    * ml	Means like constraint: require that the results have a meaning related to this string value,
    which can be any word or sequence of words. (This is effectively the reverse dictionary feature of OneLook.)
    * sl	Sounds like constraint: require that the results are pronounced similarly to this string of characters.
    (If the string of characters doesn't have a known pronunciation, the system will make its best guess using a
    text-to-phonemes algorithm.)
    * sp	Spelled like constraint: require that the results are spelled similarly to this string of characters,
    or that they match this wildcard pattern. A pattern can include any combination of alphanumeric characters, spaces,
    and two reserved characters that represent placeholders - * (which matches any number of characters) and ?
    (which matches exactly one character).
    * rel_[code]	Related word constraints: require that each of the resulting words, when paired with the word in
    this parameter, are in a predefined lexical relation indicated by [code]. Any number of these parameters may be
    specified any number of times. An assortment of semantic, phonetic, and corpus-statistics-based relations are
    available.
    [code] is a three-letter identifier from the list below.
        jja	Popular nouns modified by the given adjective, per Google Books Ngrams	gradual --> increase
        jjb	Popular adjectives used to modify the given noun, per Google Books Ngrams	beach --> sandy
        syn	Synonyms (words contained within the same WordNet synset)	ocean --> sea
        ant	Antonyms (per WordNet)	late --> early
        spc	"Kind of" (direct hypernyms, per WordNet)	gondola --> boat
        gen	"More general than" (direct hyponyms, per WordNet)	boat --> gondola
        com	"Comprises" (direct holonyms, per WordNet)	car --> accelerator
        par	"Part of" (direct meronyms, per WordNet)	trunk --> tree
        bga	Frequent followers (w' such that P(w'|w) >= 0.001, per Google Books Ngrams)	wreak --> havoc
        bgb	Frequent predecessors (w' such that P(w|w') >= 0.001, per Google Books Ngrams)	havoc --> wreak
        rhy	Rhymes ("perfect" rhymes, per RhymeZone)	spade --> aid
        nry	Approximate rhymes (per RhymeZone)	forest --> chorus
        hom	Homophones (sound-alike words)	course --> coarse
        cns	Consonant match	sample --> simple
    * v	Identifier for the vocabulary to use. If none is provided, a 550,000-term vocabulary of English words and
    multiword expressions is used. Please contact us to set up a custom vocabulary for your application.
    topics	Topic words: An optional hint to the system about the theme of the document being written.
    Results will be skewed toward these topics. At most 5 words can be specified. Space or comma delimited.
    Nouns work best.
    * lc	Left context: An optional hint to the system about the word that appears immediately to the left of the
    target word in a sentence. (At this time, only a single word may be specified.)
    * rc	Right context: An optional hint to the system about the word that appears immediately to the right of the
    target word in a sentence. (At this time, only a single word may be specified.)
    * max	Maximum number of results to return, not to exceed 1000. (default: 100)
    In the above table, the first four parameters (rd, sl, sp, rel_[code], and v) can be thought of as hard constraints
    on the result set, while the next three (topics, lc, and rc) can be thought of as context hints.
    The latter only impact the order in which results are returned. All parameters are optional.
    """

    api_root = 'https://api.datamuse.com'
    word_params = {
        'ml',
        'sl',
        'sp',
        'rel_jja',
        'rel_jjb',
        'rel_syn',
        'rel_ant',
        'rel_spc',
        'rel_gen',
        'rel_com',
        'rel_par',
        'rel_bga',
        'rel_bgb',
        'rel_rhy',
        'rel_nry',
        'rel_hom',
        'rel_cns',
        'v',
        'topics',
        'lc',
        'rc',
        'max',
    }
    suggest_params = {'s', 'max', 'v'}
    wp = Struct({k: k for k in word_params})

    def __init__(self, output_format=None):
        self.output_format = output_format

    def validate_args(self, args, param_set):
        for arg in args:
            if arg not in param_set:
                raise ValueError(
                    f'{arg} is not a valid parameter for this endpoint.'
                )

    def get_resource(self, endpoint, **kwargs):
        # I feel like this should have some kind of error handling...
        url = self.api_root + endpoint
        response = requests.get(url, params=kwargs)
        data = response.json()
        if not self.output_format:
            return data
        elif self.output_format == 'df':
            return self.dm_to_df(data)

    def words(self, **kwargs):
        self.validate_args(kwargs, self.word_params)
        words = '/words'
        return self.get_resource(words, **kwargs)

    def suggest(self, **kwargs):
        self.validate_args(kwargs, self.suggest_params)
        sug = '/sug'
        return self.get_resource(sug, **kwargs)

    @staticmethod
    def dm_to_df(datamuse_response):
        """Converts the json response of the datamuse API into a DataFrame
        :datamuse_response
            [{'word': 'foo', 'score': 100}, {'word': 'bar', 'score': 120}]
        """
        reformatted = {
            'word': [response['word'] for response in datamuse_response],
            'score': [response['score'] for response in datamuse_response],
        }
        return pd.DataFrame.from_dict(reformatted)


class DatamuseTestCase(unittest.TestCase):
    def setUp(self):
        self.api = Datamuse()
        self.max = 5

    # words endpoint
    def test_sounds_like(self):
        args = {'sl': 'orange', 'max': self.max}
        data = self.api.words(**args)
        self.assertTrue(type(data), list)
        print(('sounds like', data))

    def test_rhymes(self):
        args = {'rel_rhy': 'orange', 'max': self.max}
        data = self.api.words(**args)
        self.assertTrue(len(data) <= self.max)
        print(('rhyme', data))

    def test_near_rhymes(self):
        args = {'rel_nry': 'orange', 'max': self.max}
        data = self.api.words(**args)
        self.assertTrue(len(data) <= self.max)
        print(('near rhyme', data))

    def test_bad_request(self):
        args = {'foo': 42}
        with self.assertRaises(ValueError):
            data = self.api.words(**args)


# though really you can just run `nosetests -sv` from this directory
if __name__ == '__main__':
    unittest.main()
