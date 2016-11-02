from __future__ import division

import unittest
import requests

__author__ = 'thor'

__doc__ = """
python connector for http://www.datamuse.com/api/
Based on code from https://github.com/margaret/python-datamuse
"""

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
    'max'
}


class Datamuse():

    def __init__(self):
        self.api_root = 'https://api.datamuse.com'
        self.word_params = {
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
            'max'
        }
        self.suggest_params = {
            's',
            'max',
            'v'
        }

    def validate_args(self, args, param_set):
        for arg in args:
            if arg not in param_set:
                raise ValueError('{0} is not a valid parameter for this endpoint.'.format(arg))

    def get_resource(self, endpoint, **kwargs):
        # I feel like this should have some kind of error handling...
        url = self.api_root + endpoint
        response = requests.get(url, params=kwargs)
        data = response.json()
        return data

    def words(self, **kwargs):
        self.validate_args(kwargs, self.word_params)
        words = '/words'
        return self.get_resource(words, **kwargs)

    def suggest(self, **kwargs):
        self.validate_args(kwargs, self.suggest_params)
        sug = '/sug'
        return self.get_resource(sug, **kwargs)


def dm_to_df(datamuse_response):
    """Converts the json response of the datamuse API into a DataFrame
    :datamuse_response
        [{'word': 'foo', 'score': 100}, {'word': 'bar', 'score': 120}]
    """
    reformatted = {
        'word': [response['word'] for response in datamuse_response],
        'score': [response['score'] for response in datamuse_response]
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
        print("sounds like", data)

    def test_rhymes(self):
        args = {'rel_rhy': 'orange', 'max': self.max}
        data = self.api.words(**args)
        self.assertTrue(len(data) <= self.max)
        print("rhyme", data)

    def test_near_rhymes(self):
        args = {'rel_nry': 'orange', 'max': self.max}
        data = self.api.words(**args)
        self.assertTrue(len(data) <= self.max)
        print("near rhyme", data)

    def test_bad_request(self):
        args = {'foo': 42}
        with self.assertRaises(ValueError):
            data = self.api.words(**args)


# though really you can just run `nosetests -sv` from this directory
if __name__ == "__main__":
    unittest.main()