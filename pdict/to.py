__author__ = 'thorwhalen'

import itertools
import re
from collections import namedtuple

def namedtuple(d):
    return namedtuple('blah', d.keys())(**d)


class Struct:
    def __init__(self, obj):
        for k, v in obj.iteritems():
            if isinstance(v, dict):
                setattr(self, k, Struct(v))
            else:
                setattr(self, k, v)

    def __getitem__(self, val):
        return self.__dict__[val]

    def __repr__(self):
        return '{%s}' % str(', '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.iteritems()))


def word_replacer(rep_with_dict, inter_token_re=r"\b"):
    regex = inter_token_re + r"(?:" + "|".join(re.escape(word) for word in rep_with_dict) + r")" + inter_token_re
    reobj = re.compile(regex, re.I)
    return lambda text: reobj.sub(lambda x: rep_with_dict[x.group(0)], text)


def inverse_one_to_one(d):
    '''
    :param d: a dict that is such that each (unique) key is mapped to a unique value
    :return: returns the inverse (value->key) dict

    Example:
        inverse_one_to_one({'A':'a', 'B':'bb', 'C':'cc'})
            == {'a': 'A', 'cc': 'C', 'bb': 'B'}
    '''
    value_list = d.values()
    assert len(value_list) == len(set(value_list)), "You cannot use values_to_keys_dict() if there are duplicate values"
    return dict((v, k) for k, v in d.iteritems())


def inverse_one_to_many(d):
    '''
    :param d: a dict that is such that each (unique) key is mapped to a list of values (whose values are globally unique)
    :return: returns the inverse (value->key) dict

    Example:
        inverse_one_to_many({'A':['a','aa','aaa'], 'B':['b','bb']})
            == {'a': 'A', 'aa': 'A', 'b': 'B', 'aaa': 'A', 'bb': 'B'}
    '''
    value_list = list(itertools.chain.from_iterable(d.values()))
    assert len(value_list) == len(set(value_list)), "You cannot use values_to_keys_dict() if there are duplicate values"
    inverse_dict = dict()
    for k, v in d.iteritems():
        for vv in v:
            inverse_dict[vv] = k
    return inverse_dict


def inverse_many_to_one(d):
    '''
    :param d: a dict that is such that each (unique) key is mapped to a value, but different keys can map to the same value
    :return: returns the inverse (value->key) dict

    The keys of the inverse dict will be the unique values found in the original dict, and the values of the inverse will
    gather in the list all (original dict) keys that mapped to it.

    Example:
        inverse_many_to_one({'a': 'A', 'aa': 'A', 'b': 'B', 'aaa': 'A', 'bb': 'B'})
            == {'A':['a','aa','aaa'], 'B':['b','bb']}
    '''
    inverse_dict = {}
    for k, v in d.iteritems():
        inverse_dict[v] = inverse_dict.get(v, [])
        inverse_dict[v].append(k)
    return inverse_dict

    # value_list = list(itertools.chain.from_iterable(d.values()))
    # assert len(value_list) == len(set(value_list)), "You cannot use values_to_keys_dict() if there are duplicate values"
    # inverse_dict = dict()
    # for k, v in d.iteritems():
    #     for vv in v:
    #         inverse_dict[vv] = k
    # return inverse_dict



# def dataframe(d):
#     """
#     returns a datafame from a multi-level dict
#     NOTE: use pd.DataFrame.from_dict() instead for up to two depth levels
#     """
#     val_list
#     for key,val in d.values():
#         frames.append(pd.DataFrame.from_dict(val, orient='index'))
#     return pd.concat(frames, keys=key_list)


    # some code from someone else that words for depths of exactly 3
    # key_list = []
    # frames = []
    # for key,val in d.iteritems():
    #     key_list.append(key)
    #     # print frames
    #     frames.append(pd.DataFrame.from_dict(val, orient='index'))
    # return pd.concat(frames, keys=key_list)




# if __name__ == "__main__":









# # test for dataframe
# from pdict.to import dataframe as dict2df
# w = {12: {'Category 1': {'att_1': 1, 'att_2': 'whatever'},
#   'Category 2': {'att_1': 23, 'att_2': 'another'}},
#  15: {'Category 1': {'att_1': 10, 'att_2': 'foo'},
#   'Category 2': {'att_1': 30, 'att_2': 'bar'}}}
# df = dict2df(w)
# print w
# print df