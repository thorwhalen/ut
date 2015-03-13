__author__ = 'thorwhalen'

import pprint
import numpy as np
import pandas as pd

def typeof(x):
    if isinstance(x,list):
        if len(x) > 0:
            unik_types = list(np.lib.unique([typeof(xx) for xx in x]))
            if len(unik_types)==1:
                return "list of " + unik_types[0]
            elif len(unik_types) <= 3:
                return "list of " + ", ".join(unik_types)
            else:
                return "list of various types"
        else:
            return "empty list"
    else:
        return type(x).__name__

def print_dict_example(dict_list,recursive=True):
    ppr = pprint.PrettyPrinter(indent=2)
    ppr.pprint(example_dict_from_dict_list(dict_list,recursive=recursive))

def print_dict_example_types(dict_list,recursive=True):
    ppr = pprint.PrettyPrinter(indent=2)
    ppr.pprint(dict_of_types_of_dict_values(dict_list,recursive=recursive))

def example_dict_from_dict_list(dict_list,recursive=False):
    """
    Returns a dict that "examplifies" the input list of dicts
    Indeed, you may have a list of dicts but that don't contain the same keys.
    This function will pick the first new key,value it finds and pack it in a same dict

    If the argument recursive is True, the function will call itself recursively on any values that are lists of dicts

     For example:
        d1 = {'a':1, 'b':2}
        d2 = {'b':20, 'c':30}
        d12 = example_dict_from_dict_list([d1,d2])
        assert d12=={'a':1, 'b':2, 'c':30}
    """
    if not isinstance(dict_list,list):
        if isinstance(dict_list,dict):
            dict_list = [dict_list]
        else:
            raise TypeError("dict_list must be a dict or a list of dicts")
    else:
        if not all([isinstance(x,dict) for x in dict_list]):
            raise TypeError("dict_list must be a dict or a list of dicts")
    all_keys = set([])
    [all_keys.update(this_dict.keys()) for this_dict in dict_list] # this constructs a list of all keys encountered in the list of dicts
    example_dict = dict()
    keys_remaining_to_find = all_keys
    for this_dict in dict_list:
        new_keys = list(set(keys_remaining_to_find).intersection(this_dict.keys()))
        if not new_keys: continue
        new_dict = {k:this_dict[k] for k in new_keys if this_dict[k]} # keep only keys with non-empty and non-none value
        example_dict = dict(example_dict,**{k:v for k,v in new_dict.items()})
        keys_remaining_to_find = keys_remaining_to_find.difference(new_keys)
        if not keys_remaining_to_find: break # if there's no more keys to be found, you can quit

    if recursive==True:
        dict_list_keys = [k for k in example_dict.keys() if (k and is_dict_or_list_of_dicts(example_dict[k]))]
        for k in dict_list_keys:
            example_dict[k] = example_dict_from_dict_list(example_dict[k],recursive=True)
    return example_dict

def dict_list_key_count(dict_list):
    """
    returns a dict with all keys encoutered in the list of dicts, and values exhibiting
    how many times the key was encoutered in the dict list
    """
    all_keys = example_dict_from_dict_list(dict_list).keys()
    return {k:np.sum(np.array([d.has_key(k) for d in dict_list])) for k in all_keys}

def dict_list_has_key_df(dict_list,index_names=None,use_0_1=False):
    """
    returns a dataframe where:
        * indices indicate what dict_list element the row corresponds to,
        * columns are all keys ever encoutered in the list of dicts, and
        * df[i,j] is True if dict i has key j
    """
    df = pd.concat([pd.Series({k:True for k in d.keys()}) for d in dict_list],axis=1).transpose()
    df.fillna(False,inplace=True)
    if use_0_1==True:
        df.replace([True,False],[1,0],inplace=True)
    if index_names:
        df.index = index_names
    return df


def dict_of_types_of_dict_values(x,recursive=False):
    if isinstance(x,list): # if x is a list of dicts
        x = example_dict_from_dict_list(x,recursive=True)
    if recursive==False:
        return {k:typeof(x[k])for k in x.keys()}
    else:
        dict_of_types = dict()
        for k in x.keys():
            if isinstance(x[k],dict):
                dict_of_types = dict(dict_of_types,**{k:{'dict':dict_of_types_of_dict_values(x[k],recursive=True)}})
            elif is_list_of_dicts(x[k]):
                dict_of_types = dict(dict_of_types,**{k:{'list of dict':dict_of_types_of_dict_values(x[k],recursive=True)}})
            else:
                dict_of_types = dict(dict_of_types,**{k:typeof(x[k])})
        return dict_of_types

def is_list_of_dicts(x):
    return isinstance(x,list) and len(x)>0 and all([isinstance(xx,dict) for xx in x])

def is_dict_or_list_of_dicts(x):
    return isinstance(x,dict) or is_list_of_dicts(x)


# test_example_dict_from_dict_list()
# d1 = {'a':1, 'b':2}
# d2 = {'b':20, 'c':30}
# d12 = pdict.example_dict_from_dict_list([d1,d2])
# print d12
# print {'a':1, 'b':2, 'c':30}
# d12=={'a':1, 'b':2, 'c':30}

if __name__ == "__main__":
    t = {'a':[{
                  'aa':[1,2,3],
                  'bb':[
                      {'bba':2,'bbb':3},
                      {'bba2':[1,2,3],'bbb2':'boo'},
                      {'bba3':{'dict':'this is a dict'}}]
              },
              {'aa2':[2,3,4]}],
         'b':[4,5,6]}
    print_dict_example_types(t,recursive=True)
