"""Utils to work with python objects"""

__author__ = 'thor'

import ut.pdict.get as pdict_get
import ut.util.ulist as util_ulist
from optparse import OptionParser
import inspect
import pickle
import zlib
import types

function_type = type(lambda x: x)  # using this instead of callable() because classes are callable, for instance


def copy_attrs(target, source, attrs, raise_error_if_an_attr_is_missing=True):
    """Copy attributes from one object to another.
    >>> class A:
    ...     x = 0
    >>> class B:
    ...     x = 1
    ...     yy = 2
    ...     zzz = 3
    >>> dict_of = lambda o: {a: getattr(o, a) for a in dir(A) if not a.startswith('_')}
    >>> dict_of(A)
    {'x': 0}
    >>> copy_attrs(A, B, 'yy')
    >>> dict_of(A)
    {'x': 0, 'yy': 2}
    >>> copy_attrs(A, B, ['x', 'zzz'])
    >>> dict_of(A)
    {'x': 1, 'yy': 2, 'zzz': 3}

    But if you try to copy something that `B` (the source) doesn't have, copy_attrs will complain:
    >>> copy_attrs(A, B, 'this_is_not_an_attr')
    Traceback (most recent call last):
        ...
    AttributeError: type object 'B' has no attribute 'this_is_not_an_attr'

    If you tell it not to complain, it'll just ignore attributes that are not in source.
    >>> copy_attrs(A, B, ['nothing', 'here', 'exists'], raise_error_if_an_attr_is_missing=False)
    >>> dict_of(A)
    {'x': 1, 'yy': 2, 'zzz': 3}
    """
    if isinstance(attrs, str):
        attrs = (attrs,)
    if raise_error_if_an_attr_is_missing:
        filt = lambda a: True
    else:
        filt = lambda a: hasattr(source, a)
    for a in filter(filt, attrs):
        setattr(target, a, getattr(source, a))


def is_classmethod(class_, attr):
    attr = getattr(class_, attr)
    return inspect.ismethod(attr) and getattr(attr, "__self__") == class_


def list_of_properties_instancemethods_and_classmethods_for_class(class_):
    props = list()
    instance_methods = list()
    class_methods = list()

    for attr_str in (x for x in list(class_.__dict__.keys()) if not x.startswith('__')):
        attr = getattr(class_, attr_str)
        if inspect.ismethod(attr):
            if getattr(attr, "__self__") == class_:
                class_methods.append(attr_str)
            else:
                instance_methods.append(attr_str)
        else:
            props.append(attr_str)

    return props, instance_methods, class_methods


def list_of_properties_instancemethods_and_classmethods_for_obj(obj):
    props = list()
    instance_methods = list()
    class_methods = list()

    for attr_str in (x for x in list(obj.__dict__.keys()) if not x.startswith('__')):
        attr = getattr(obj, attr_str)
        if inspect.ismethod(attr):
            # if getattr(attr, "__self__") == obj.__class__:
            #     class_methods.append(attr_str)
            # else:
            instance_methods.append(attr_str)
        else:
            props.append(attr_str)
    _, _, class_methods = list_of_properties_instancemethods_and_classmethods_for_class(obj.__class__)

    return props, instance_methods, class_methods


def zpickle_dumps(obj):
    return zlib.compress(pickle.dumps(obj))


def zpickle_loads(zpickle_string):
    return pickle.loads(zlib.decompress(zpickle_string))


def inject_method(self, method_function, method_name=None):
    if isinstance(method_function, function_type):
        if method_name is None:
            method_name = method_function.__name__
        setattr(self,
                method_name,
                types.MethodType(method_function, self))
    else:
        if isinstance(method_function, dict):
            method_function = [(func, func_name) for func_name, func in method_function.items()]
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


def set_attributes(obj, attr_dict=None, default_attr_dict=None):
    '''
    Setting attributes and values (specified by a dict) to an object instance, possibly completing the attributes with
    defaults.
    :param obj: an object
    :param attr_dict: dict of (attr, val) pairs to assign
    :param default_attr_dict: (optional) default attr_dict to complete attr_dict with
    :return:
    '''
    # if default_attributes were given, complete attr_dict with them
    if attr_dict is None:
        attr_dict = dict()
    if default_attr_dict:
        attr_dict = pdict_get.left_union(attr_dict, default_attr_dict)
    # loop through attr_dict and assign attributes to obj
    for k, v in attr_dict.items():
        setattr(obj, k, v)
    # return obj
    return obj


def has_attributes(obj, attr_list):
    attr_list = util_ulist.ascertain_list(attr_list)
    return all([x in list(obj.__dict__.keys()) for x in attr_list])


def has_callable_attr(obj, attr):
    return hasattr(obj, attr) and hasattr(getattr(obj, attr), '__call__')


def has_non_callable_attr(obj, attr):
    return hasattr(obj, attr) and not hasattr(getattr(obj, attr), '__call__')
