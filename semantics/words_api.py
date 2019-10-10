import requests
from pprint import pprint
import os

__doc__ = """ Copied from https://github.com/ehwhydubya/pywords (MIT licence)"""

class Words(object):
    """A class that interfaces with the Words API at https://wordsapi.com"""

    pretty_print_error = ValueError(
        'You must enter \'on\' or \'off\' to toggle the pretty print setting. Please enter a valid input.')
    detail_error = ValueError(
        'You must choose a valid detail. See https://www.wordsapi.com/docs#details for valid details.')
    entry_error = ValueError('You must enter a valid string')

    _details = {
        'definitions': '{}/definitions',
        'synonyms': '{}/synonyms',
        'antonyms': '{}/antonyms',
        'examples': '{}/examples',
        'typeOf': '{}/typeOf',
        'hasTypes': '{}/hasTypes',
        'partOf': '{}/partOf',
        'hasParts': '{}/hasParts',
        'instanceOf': '{}/instanceOf',
        'hasInstances': '{}/hasInstances',
        'similarTo': '{}/similarTo',
        'also': '{}/also',
        'entails': '{}/entails',
        'memberOf': '{}/memberOf',
        'hasMembers': '{}/hasMembers',
        'substanceOf': '{}/substanceOf',
        'hasSubstances': '{}/hasSubstances',
        'inCategory': '{}/inCategory',
        'hasCategories': '{}/hasCategories',
        'usageOf': '{}/usageOf',
        'hasUsages': '{}/hasUsages',
        'inRegion': '{}/inRegion',
        'regionOf': '{}/regionOf',
        'pertainsTo': '{}/pertainsTo',
        'rhymes': '{}/rhymes',
        'frequency': '{}/frequency'
    }

    def __init__(self, api_key=None, pretty=None):
        if api_key is None:
            if 'WORDSAPI_KEY' not in os.environ:
                print("You need an api key to use this. Get it at https://www.mashape.com/")
                print("At the time of this writing, exactly here: https://market.mashape.com/wordsapi/wordsapi")
                print("I suggest you enter this key in an environmental variable under the name WORDSAPI_KEY")
                print("Then, you won't need to specify it every time")
                raise ValueError("No api_key given or found in environmental variable name WORDSAPI_KEY")
            else:
                api_key = os.getenv('WORDSAPI_KEY')
        self._base_url = 'https://wordsapiv1.p.mashape.com/words/'
        self._auth_headers = {'X-Mashape-Key': api_key}  # auth against Mashape APIs
        self.pretty_print = pretty  # pretty print set to False by default for functionality, can be set to True for pretty one-offs

    def setPrettyPrint(self, toggle):
        if toggle == 'on':
            self.pretty_print = True
        elif toggle == 'off':
            self.pretty_print = False
        else:
            raise Words.pretty_print_error

    def isPrettyPrint(self):
        if self.pretty_print:
            return True
        else:
            return False

    def _get(self, word, detail):

        # if word is not type(str):
        #     raise Words.entry_error
        if not isinstance(word, str):
            raise Words.entry_error

        if detail not in Words._details:
            raise Words.detail_error

        url = self._base_url + Words._details[detail].format(word)
        query = requests.get(url, headers=self._auth_headers)

        if self.isPrettyPrint():
            return pprint(query.json())
        else:
            return query.json()

    def random(self):
        '''
        Returns a random word. No arguments needed.
        '''
        query = requests.get(self._base_url, params={'random': 'true'}, headers=self._auth_headers)

        if self.isPrettyPrint():
            return pprint(query.json())
        else:
            return query.json()

    def word(self, word):
        '''
        Retrieves everything that the Words API has on a word.
        '''
        url = self._base_url + word
        query = requests.get(url, headers=self._auth_headers)

        if self.isPrettyPrint():
            return pprint(query.json())
        else:
            return query.json()

    def definitions(self, word):
        '''
        The meaning of the word, including its part of speech.
        See https://www.wordsapi.com/docs#words for more info.
        '''
        return self._get(word, 'definitions')

    def synonyms(self, word):
        '''
        Words that can be interchanged for the original word in the same context.
        '''
        return self._get(word, 'synonyms')

    def antonyms(self, word):
        '''
        Words that have the opposite context of the original word.
        '''
        return self._get(word, 'antonyms')

    def examples(self, word):
        '''
        Example sentences using the word.
        '''
        return self._get(word, 'examples')

    def typeOf(self, word):
        '''
        Words that are more generic than the original word. Also known as hypernyms.
        For example, a hatchback is a type of car.
        '''
        return self._get(word, 'typeOf')

    def hasTypes(self, word):
        '''
        Words that are more specific than the original word. Also known as hyponyms.
        For example, purple has types violet, lavender, mauve, etc.
        '''
        return self._get(word, 'hasTypes')

    def partOf(self, word):
        '''
        The larger whole to which this word belongs. Also known as holonyms.
        For example, a finger is part of a hand, a glove, a paw, etc.
        '''
        return self._get(word, 'partOf')

    def hasParts(self, word):
        '''
        Words that are part of the original word. Also known as meronyms.
        For example, a building has parts such as roofing, plumbing etc.
        '''
        return self._get(word, 'hasParts')

    def instanceOf(self, word):
        '''
        Words that the original word is an example of.
        For example, Einstein is an instance of a physicist.
        '''
        return self._get(word, 'instanceOf')

    def hasInstances(self, word):
        '''
        Words that are examples of the original word.
        For example, president has instances such as theodore roosevelt, van buren, etc.
        '''
        return self._get(word, 'hasInstances')

    def similarTo(self, word):
        '''
        Words that similar to the original word, but are not synonyms.
        For example, red is similar to bloody.
        '''
        return self._get(word, 'similarTo')

    def also(self, word):
        '''
        Phrases to which the original word belongs.
        For example, bump is used in the phrase bump off.
        '''
        return self._get(word, 'also')

    def entails(self, word):
        '''
        Words that are implied by the original word. Usually used for verbs.
        For example, rub entails touch.
        '''
        return self._get(word, 'entails')

    def memberOf(self, word):
        '''
        A group to which the original word belongs.
        For example, dory is a member of the family zeidae.
        '''
        return self._get(word, 'memberOf')

    def hasMembers(self, word):
        '''
        Words that belong to the group defined by the original word.
        For example, a cult has members called cultists.
        '''
        return self._get(word, 'hasMembers')

    def substanceOf(self, word):
        '''
        Substances to which the original word is a part of.
        For example, water is a substance of sweat.
        '''
        return self._get(word, 'substanceOf')

    def hasSubstance(self, word):
        '''
        Substances that are part of the original word.
        For example, wood has a substance called lignin.
        '''
        return self._get(word, 'hasSubstance')

    def inCategory(self, word):
        '''
        The domain category to which the original word belongs.
        For example, chaotic is in category physics.
        '''
        return self._get(word, 'inCategory')

    def hasCategories(self, word):
        '''
        Categories of the original word.
        For example, math has categories such as algebra, imaginary, numerical analysis, etc.
        '''
        return self._get(word, 'hasCategories')

    def usageOf(self, word):
        '''
        Words that the original word is a domain usage of.
        For example, advil is a useage of the trademark, etc.
        '''
        return self._get(word, 'usageOf')

    def hasUsages(self, word):
        '''
        Words that are examples of the domain the original word defines.
        For example, colloquialism is a domain that includes examples like big deal, blue moon, etc.
        '''
        return self._get(word, 'hasUsages')

    def inRegion(self, word):
        '''
        Regions where the word is used.
        For example, chips is used in region Britain.
        '''
        return self._get(word, 'inRegion')

    def regionOf(self, word):
        '''
        A region where words are used.
        For example, Canada is the region of pogey.
        '''
        return self._get(word, 'regionOf')

    def pertainsTo(self, word):
        '''
        Words to which the original word is relevant
        For example, .22-caliber pertains to caliber.
        '''
        return self._get(word, 'pertainsTo')

    def rhymes(self, word):
        '''
        See https://www.wordsapi.com/docs#rhymes
        '''
        return self._get(word, 'rhymes')

    def frequency(self, word):
        '''
        See https://www.wordsapi.com/docs#frequency
        '''
        return self._get(word, 'frequency')

    def search(self, **kwargs):
        '''
        See https://www.wordsapi.com/docs#search
        '''
        query = requests.get(self._base_url, params=kwargs, headers=self._auth_headers)

        if self.isPrettyPrint():
            return pprint(query.json())
        else:
            return query.json()