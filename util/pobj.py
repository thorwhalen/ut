__author__ = 'thor'

import ut.pdict.get as pdict_get
import ut.util.ulist as util_ulist


def set_attributes(obj, attr_dict=dict(), default_attr_dict=None):
    '''
    Setting attributes and values (specified by a dict) to an object instance, possibly completing the attributes with
    defaults.
    :param obj: an object
    :param attr_dict: dict of (attr, val) pairs to assign
    :param default_attr_dict: (optional) default attr_dict to complete attr_dict with
    :return:
    '''
    # if default_attributes were given, complete attr_dict with them
    if default_attr_dict:
        attr_dict = pdict_get.left_union(attr_dict, default_attr_dict)
    # loop through attr_dict and assign attributes to obj
    for k, v in attr_dict.iteritems():
        setattr(obj, k, v)
    # return obj
    return obj

def has_attributes(obj, attr_list):
    attr_list = util_ulist.ascertain_list(attr_list)
    return all([x in obj.__dict__.keys() for x in attr_list])


def has_callable_attr(obj, attr):
    return hasattr(obj, attr) and hasattr(getattr(obj, attr), '__call__')

def has_non_callable_attr(obj, attr):
    return hasattr(obj, attr) and not hasattr(getattr(obj, attr), '__call__')