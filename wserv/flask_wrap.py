"""Automatically making flask webservices from python objects"""
from numpy import ndarray
from pandas import Series, DataFrame
import re
import json

from ut.others.lru_cache import lru_cache
# from werkzeug.exceptions import InternalServerError

import ut.wserv.errors as err

# args_info = defaultdict(lambda: {'type': str, 'default': None})
# args_info['host'] = {'type': str, 'default': DEFAULT_SDACC_HOST}
# args_info['db'] = {'type': str, 'default': DEFAULT_SDACC_DB}
# args_info['corpus'] = {'type': str, 'default': DEFAULT_SDACC_CORPUS}

DFLT_RESULT_FIELD = 'result'


def get_pattern_from_attr_permissions_dict(attr_permissions):
    """
    Construct a compiled regular expression from a permissions dict containing a list of what to include and exclude.
    Will be used in ObjWrapper if permissible_attr_pattern is a dict.
    Note that the function enforces certain patterns (like inclusions ending with $ unless they end with *, etc.
    What is not checked for is that the "." was meant, or if it was "\." that was meant.
    This shouldn't be a problem in most cases, and hey! It's to the user to know regular expressions!
    :param attr_permissions: A dict of the format {'include': INCLUSION_LIST, 'exclude': EXCLUSION_LIST}.
        Both 'include' and 'exclude' are optional, and their lists can be empty.
    :return: a re.compile object

    >>> attr_permissions = {
    ...     'include': ['i.want.this', 'he.wants.that'],
    ...     'exclude': ['i.want', 'he.wants', 'and.definitely.not.this']
    ... }
    >>> r = get_pattern_from_attr_permissions_dict(attr_permissions)
    >>> test = ['i.want.this', 'i.want.this.too', 'he.wants.that', 'he.wants.that.other.thing',
    ...         'i.want.ice.cream', 'he.wants.me'
    ...        ]
    >>> for t in test:
    ...     print("{}: {}".format(t, bool(r.match(t))))
    i.want.this: True
    i.want.this.too: False
    he.wants.that: True
    he.wants.that.other.thing: False
    i.want.ice.cream: False
    he.wants.me: False
    """

    s = ""

    # process inclusions
    corrected_list = []
    for include in attr_permissions.get('include', []):
        if not include.endswith('*'):
            if not include.endswith('$'):
                include += '$'
        else:  # ends with "*"
            if include.endswith('\.*'):
                # assume that's not what the user meant, so change
                include = include[:-3] + '.*'
            elif include[-2] != '.':
                # assume that's not what the user meant, so change
                include = include[:-1] + '.*'
        corrected_list.append(include)
    s += '|'.join(corrected_list)

    # process exclusions
    corrected_list = []
    for exclude in attr_permissions.get('exclude', []):
        if not exclude.endswith('$') and not exclude.endswith('*'):
            # add to exclude all subpaths if not explicitly ending with "$"
            exclude += '.*'
        else:  # ends with "*"
            if exclude.endswith('\.*'):
                # assume that's not what the user meant, so change
                exclude = exclude[:-3] + '.*'
            elif exclude[-2] != '.':
                # assume that's not what the user meant, so change
                exclude = exclude[:-1] + '.*'
        corrected_list.append(exclude)
    if corrected_list:
        s += '(?!' + '|'.join(corrected_list) + ')'

    return re.compile(s)


def default_to_jdict(result, result_field=DFLT_RESULT_FIELD):
    if isinstance(result, list):
        return {result_field: result}
    elif isinstance(result, ndarray):
        return {result_field: result.tolist()}
    elif isinstance(result, dict) and len(result) > 0:
        first_key, first_val = next(iter(result.items()))  # look at the first key to determine what to do with the dict
        if isinstance(first_key, int):
            key_trans = chr
        else:
            key_trans = lambda x: x
        if isinstance(first_val, ndarray):
            return {result_field: {key_trans(k): v.tolist() for k, v in result.items()}}
        elif isinstance(first_val, dict):
            return {result_field: {key_trans(k): default_to_jdict(v) for k, v in result.items()}}
        else:
            return {key_trans(k): v for k, v in result.items()}
    elif isinstance(result, (Series, DataFrame)):
        return json.loads(result.to_json())
        # return default_to_jdict(result.to_dict())
    elif hasattr(result, 'to_json'):
        return json.loads(result.to_json())
    else:
        try:
            return {result_field: result}
        except TypeError:
            if hasattr(result, 'next'):
                return {result_field: list(result)}
            else:
                return {result_field: str(result)}


def extract_kwargs(request, convert_arg=None, file_var='file'):
    if convert_arg is None:
        convert_arg = {}
    kwargs = dict()
    for k in list(request.args.keys()):
        if k in convert_arg:
            if 'default' in convert_arg[k]:
                kwargs[k] = request.args.get(k,
                                             type=convert_arg[k].get('type', str),
                                             default=convert_arg[k]['default'])
            else:
                kwargs[k] = request.args.get(k, type=convert_arg[k].get('type', str))
        else:
            kwargs[k] = request.args.get(k)
    if request.json is not None:
        for k, v in request.json.items():
            if k in convert_arg:
                _type = convert_arg[k].get('type', None)
                if callable(_type):
                    v = _type(v)
            kwargs[k] = v
    if 'file' in request.files:
        kwargs[file_var] = request.files['file']
    return kwargs


class ObjWrapper(object):
    def __init__(self,
                 obj_constructor,
                 obj_constructor_arg_names=None,  # used to determine the params of the object constructors
                 convert_arg=None,  # input processing: Dict specifying how to prepare ws arguments for methods
                 file_var='file',  # input processing: name of the variable to use if there's a 'file' in request.files
                 permissible_attr_pattern='[^_].*',  # what attributes are allowed to be accessed
                 to_jdict=default_to_jdict,  # output processing: Function to convert an output to a jsonizable dict
                 obj_str='obj',  # name of object to use in error messages
                 cache_size=5,
                 debug=0):
        """
        An class to wrap a "controller" class for a web service API.
        It takes care of LRU caching objects constructed before (so they don't need to be re-constructed for every
        API call), and converting request.json and request.args arguments to the types that will be recognized by
        the method calls.
        :param obj_constructor: a function that, given some arguments, constructs an object. It is this object
            that will be wrapped for the webservice
        :param obj_constructor_arg_names:
        :param convert_arg: (processing) a dict keyed by variable names (str) and valued by a dict containing a
            'type': a function (typically int, float, bool, and list) that will convert the value of the variable
                to make it web service compliant
            'default': A value to assign to the variable if it's missing.
        :param The pattern that determines what attributes are allowed to be accessed. Note that patterns must be
            complete patterns (i.e. describing the whole attribute path, not just a subset. For example, if you want
            to have access to this.given.thing you, specifying r"this\.given" won't be enough. You need to specify
            "this\.given.thing" or "this\.given\..*" (the latter giving access to all children of this.given.).
            Allowed formats:
                a re.compiled pattern
                a string (that will be passed on to re.compile()
                a dict with either
                    an "exclude", pointing to a list of patterns to exclude
                    an "include", pointing to a list of patterns to include
        :param to_jdict: (input processing) Function to convert an output to a jsonizable dict
        """
        self.obj_constructor = lru_cache(maxsize=cache_size)(obj_constructor)

        if obj_constructor_arg_names is None:
            obj_constructor_arg_names = []
        elif isinstance(obj_constructor_arg_names, str):
            obj_constructor_arg_names = [obj_constructor_arg_names]
        self.obj_constructor_arg_names = obj_constructor_arg_names

        if convert_arg is None:
            convert_arg = {}
        self.convert_arg = convert_arg  # a specification of how to convert specific argument names or types
        self.file_var = file_var

        if isinstance(permissible_attr_pattern, dict):
            self.permissible_attr_pattern = get_pattern_from_attr_permissions_dict(permissible_attr_pattern)
        else:
            self.permissible_attr_pattern = re.compile(permissible_attr_pattern)
        self.to_jdict = to_jdict
        self.obj_str = obj_str
        self.debug = debug

    def _get_kwargs_from_request(self, request):
        """
        Translate the flask request object into a dict, taking first the contents of request.arg,
        converting them to a type if the name is listed in the convert_arg property, and assigning a default value
        (if specified bu convert_arg), and then updating with the contents of request.json
        :param request: the flask request object
        :return: a dict of kwargs corresponding to the union of post and get arguments
        """
        kwargs = extract_kwargs(request, convert_arg=self.convert_arg, file_var=self.file_var)

        return dict(kwargs)

    def _is_permissible_attr(self, attr):
        return bool(self.permissible_attr_pattern.match(attr))

    def obj(self, obj, attr=None, result_field=DFLT_RESULT_FIELD, **method_kwargs):
        # get or make the root object
        if isinstance(obj, dict):
            obj = self.obj_constructor(**obj)
        elif isinstance(obj, (tuple, list)):
            obj = self.obj_constructor(*obj)
        elif obj is not None:
            obj = self.obj_constructor(obj)
        else:
            obj = self.obj_constructor()

        # at this point obj is an actual obj_constructor constructed object....

        # get the leaf object
        if attr is None:
            raise err.MissingAttribute()
        obj = get_attr_recursively(obj, attr)  # at this point obj is the nested attribute object

        # call a method or return a property
        if callable(obj):
            return self.to_jdict(obj(**method_kwargs), result_field=result_field)
        else:
            return self.to_jdict(obj, result_field=result_field)

    def robj(self, request):
        """
        Translates a request to an object access (get property value or call object method).
            Uses self._get_kwargs_from_request(request) to get a dict of kwargs from request.arg
        (with self.convert_arg conversions)
        and request.json.
            The object to be constructed (or retrieved from cache) is determined by the self.obj_constructor_arg_names
        list. The names listed there will be extracted from the request kwargs and passed on to the object constructor
        (or cache).
            The attribute (property or method) to be accessed is determined by the 'attr' argument, which is a
        period-separated string specifying the path to the attribute (e.g. "this.is.what.i.want" will access
        obj.this.is.what.i.want).
            A requested attribute is first checked against "self._is_permissible_attr" before going further.
        The latter method is used to control access to object attributes.
        :param request: A flask request
        :return: The value of an object's property, or the output of a method.
        """
        kwargs = self._get_kwargs_from_request(request)
        if self.debug > 0:
            print(("robj: kwargs = {}".format(kwargs)))
        obj_kwargs = {k: kwargs.pop(k) for k in self.obj_constructor_arg_names if k in kwargs}

        attr = kwargs.pop('attr', None)
        if attr is None:
            raise err.MissingAttribute()
        elif not self._is_permissible_attr(attr):
            print(attr)
            print(str(self.permissible_attr_pattern.pattern))
            raise err.ForbiddenAttribute(attr)
        if self.debug > 0:
            print(("robj: attr={}, obj_kwargs = {}, kwargs = {}".format(attr, obj_kwargs, kwargs)))
        return self.obj(obj=obj_kwargs, attr=attr, **kwargs)


def get_attr_recursively(obj, attr, default=None):
    try:
        for attr_str in attr.split('.'):
            obj = getattr(obj, attr_str)
        return obj
    except AttributeError:
        return default


def obj_str_from_obj(obj):
    try:
        return obj.__class__.__name__
    except AttributeError:
        return 'obj'

# Will delete another day. Just spent two hours making this beautiful and useful object, and then realized
# it had a name, and code for it (is even integrated in the standard libraries of Python 3+
# It's called lru_cache.
#
# class CachedObjAccess(object):
#     """
#     This class allows you to not have to re-construct objects every time you need them.
#      It works by caching the objects you've asked for before, and giving you the already constructed objects from
#      the cache itself.
#     It's basically a constructor memoizer that (1) has a limited cache and (2) gets rid of the cache entry that
#     was accessed the longest time ago if the cache is at capacity when a new object needs to be inserted.
#
#     https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_.28LRU.29
#
#     """
#     def __init__(self,
#                  obj_constructor,
#                  constructor_kwargs_for_key,
#                  cache_size = 10,
#                  ckey_from_key=lambda x: x):
#         """
#
#         :param obj_constructor: The function (or class constructor) that constructs the objects.
#         :param constructor_kwargs_for_key: A function that
#         :param cache_size:
#         :param ckey_from_key:
#         """
#         self.obj_constructor = obj_constructor
#         self.constructor_kwargs_for_key = constructor_kwargs_for_key
#         self.cache_size = cache_size
#         self.ckey_from_key = ckey_from_key
#         self.cobj_for_ckey = dict()  # the cache!
#
#     def get_obj(self, key):
#         """
#         Get the obj for that key, if it's in the cache (cobj_for_ckey), and if it's not,
#         construct it (with obj_constructor(**constructor_kwargs_for_key(key)) and insert it in the cache,
#         removing the oldest (accessed the longest time ago) entry if the cache is at capacity
#         (i.e. len(cobj_for_ckey) >= cache_size
#         :param key:
#         :return:
#         """
#         ckey = self.ckey_from_key(key)
#         cobj = self.cobj_for_ckey.get(ckey, None)
#         if cobj is not None:
#             cobj['last_accessed_utc'] = datetime.utcnow()
#             return cobj['obj']
#         else:
#             cobj = {'obj': self.obj_constructor(**self.constructor_kwargs_for_key(key)),
#                     'last_accessed_utc': datetime.utcnow()}
#             self.insert_cobj(ckey, cobj)
#             return cobj['obj']
#
#     def insert_cobj(self, ckey, cobj):
#         if len(self.cobj_for_ckey) >= self.cache_size:  # if we're at the limit, make some space...
#             # find the ckey with the minimum last_accessed_utc (the ckey that's been accessed the longest time ago)
#             min_last_accessed_utc = cobj['last_accessed_utc']
#             pop_ckey = None
#             for _ckey, _cobj in self.cobj_for_ckey.iteritems():
#                 if _cobj['last_accessed_utc'] < min_last_accessed_utc:
#                     min_last_accessed_utc = _cobj['last_accessed_utc']
#                     pop_ckey = _ckey
#
#             # pop the ckey that's been accessed the longest time ago
#             if pop_ckey is not None:
#                 self.cobj_for_ckey.pop(pop_ckey)
#
#         # add the {ckey: cobj} entry
#         self.cobj_for_ckey[ckey] = cobj
#
