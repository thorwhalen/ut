__author__ = 'thor'

import ut.pdict.get as pdict_get
import ut.util.ulist as util_ulist
from optparse import OptionParser
import inspect
import cPickle
import zlib
import types
function_type = type(lambda x: x)  # using this instead of callable() because classes are callable, for instance


def zpickle_dumps(obj):
    return zlib.compress(cPickle.dumps(obj))


def zpickle_loads(zpickle_string):
    return cPickle.loads(zlib.decompress(zpickle_string))


def inject_method(self, method_function, method_name=None):
    if isinstance(method_function, function_type):
        if method_name is None:
            method_name = method_function.__name__
        setattr(self,
                method_name,
                types.MethodType(method_function, self))
    else:
        if isinstance(method_function, dict):
            method_function = [(func, func_name) for func_name, func in method_function.iteritems()]
        for method in method_function:
            if isinstance(method, tuple) and len(method) == 2:
                self = inject_method(self, method[0], method[1])
            else:
                self = inject_method(self, method)

    return self


def methods_of(obj_or_class):
    if type(obj_or_class) == type:
        t = inspect.getmembers(obj_or_class, predicate=inspect.ismethod)
    else:
        t = inspect.getmembers(obj_or_class.__class__, predicate=inspect.ismethod)
    return [x[0] for x in t]


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



