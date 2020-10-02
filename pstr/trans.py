__author__ = 'thorwhalen'

from unidecode import unidecode
import re
import pandas as pd

########### Partial and incremental formatting #########################################################################
from string import Formatter


# Note: Some (or all) of this might not be necessary if we use an ini format and parser supporting reference following
class PartialFormatter(Formatter):
    """A string formatter that won't complain if the fields are only partially formatted.
    But note that you will lose the spec part of your template (e.g. in {foo:1.2f}, you'll loose the 1.2f
    if not foo is given -- but {foo} will remain).
    """

    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except KeyError:
            return '{' + key + '}'

    def format_fields_set(self, s):
        return {x[1] for x in self.parse(s) if x[1]}


partial_formatter = PartialFormatter()


# TODO: For those who love algorithmic optimization, there's some wasted to cut out here below.

def _unformatted(d):
    for k, v in d.items():
        if isinstance(v, str) and len(partial_formatter.format_fields_set(v)) > 0:
            yield k


def _fields_to_format(d):
    for k, v in d.items():
        if isinstance(v, str):
            yield from partial_formatter.format_fields_set(v)


def format_str_vals_of_dict(d, *, max_formatting_loops=10, **kwargs):
    """Recursively replaces format strings of a dict (whose keys and values are strings) with values taken from **kwargs

    :param d: Dict whos values we should replace
    :param max_formatting_loops:
    :param kwargs: key-value pairs defining replacement: All instances of {key} will be replaced with value
    :return:

    >>> d = {
    ...     'filepath': '{root}/{file}.{ext}',
    ...     'ext': 'txt'
    ... }
    >>> format_str_vals_of_dict(d, root='ROOT', file='FILE')
    {'filepath': 'ROOT/FILE.txt', 'ext': 'txt'}

    Note that if the input mapping `d` and the kwargs have a conflict, the mapping version is used!

    >>> format_str_vals_of_dict(d, root='ROOT', file='FILE', ext='will_not_be_used')
    {'filepath': 'ROOT/FILE.txt', 'ext': 'txt'}

    But if you want to override an input mapping, you can -- the usual way:
    >>> format_str_vals_of_dict(dict(d, ext='will_be_used'), root='ROOT', file='FILE')
    {'filepath': 'ROOT/FILE.will_be_used', 'ext': 'will_be_used'}

    If you don't provide enough fields to satisfy all the format fields in the values of `d`,
    you'll be told to bugger off.

    >>> format_str_vals_of_dict(d, root='ROOT')
    Traceback (most recent call last):
    ...
    ValueError: I won't be able to complete that. You'll need to provide the values for:
      file

    And it's recursive...
    >>> d = {
    ...     'filepath': '{root}/{filename}',
    ...     'filename': '{file}.{ext}'
    ... }
    >>> my_configs = {'root': 'ROOT', 'file': 'FILE', 'ext': 'EXT'}
    >>> format_str_vals_of_dict(d, **my_configs)
    {'filepath': 'ROOT/FILE.EXT', 'filename': 'FILE.EXT'}

    # TODO: Could make the above work if filename is give, but not file nor ext! At least as an option.

    """
    d = dict(**d)  # make a shallow copy
    # The defaults (kwargs) cannot overlap with any keys of d, so:
    kwargs = {k: kwargs[k] for k in set(kwargs) - set(d)}

    provided_fields = set(d) | set(kwargs)
    missing_fields = set(_fields_to_format(d)) - provided_fields

    if missing_fields:
        raise ValueError("I won't be able to complete that. You'll need to provide the values for:\n" +
                         f"  {', '.join(missing_fields)}")

    for i in range(max_formatting_loops):
        unformatted = set(_unformatted(d))

        if unformatted:
            for k in unformatted:
                d[k] = partial_formatter.format(d[k], **kwargs, **d)
        else:
            break
    else:
        raise ValueError(f"There are still some unformatted fields, "
                         f"but I reached my max {max_formatting_loops} allowed loops. " +
                         f"Those fields are: {set(_fields_to_format(d)) - (set(d) | set(kwargs))}")

    return d


multiple_spaces_exp = re.compile('\s\s*')


def pycli_of_str(s):
    """

    :param s: a string assumed to be python code
    :return: a string that would correspond to this code written in a python cli (you know, with the >>> and ...)
    """
    ss = ''
    for line in s.split('\n'):
        if len(line) == 0:
            ss += '>>> ' + line + '\n'
        elif line[0].isspace() and line[0] != '\n':
            ss += '...' + line + '\n'
        else:
            ss += '>>> ' + line + '\n'
    return ss


def indent_string_block(s, indent=2):
    """
    Indents all the lines of s by indent number of spaces
    """
    indent_str = ' ' * indent
    return indent_str + s.replace('\n', '\n' + indent_str)


def str_to_unicode_or_bust(obj, encoding='utf-8'):
    if isinstance(obj, str):
        if not isinstance(obj, str):
            obj = str(obj, encoding)
    return obj


def str_to_utf8_or_bust(obj):
    if isinstance(obj, str):
        try:
            obj = obj.encode('utf-8')
        except UnicodeDecodeError:
            obj = obj.decode('ISO-8859-1').encode('utf-8')
    return obj


def to_utf8_or_bust_iter(it):
    return it.map(str_to_utf8_or_bust)


def to_unicode_or_bust(obj, encoding='utf-8'):
    try:
        if isinstance(obj, str):
            if not isinstance(obj, str):
                obj = str(obj, encoding)
        else:
            if isinstance(obj, pd.DataFrame):
                for c in obj.columns:
                    obj[c] = to_unicode_or_bust(obj[c], encoding)
            else:
                try:
                    obj = [str(x, encoding) for x in obj]
                except:
                    pass
        return obj
        # print "changed something"
        # print type(obj)
    except:
        UnicodeError("to_unicode_or_bust failed with %s" % obj)


# at some point to_unicode_or_bust looked like follows, but didn't make sense (and had bugs, so I did the above)
# def to_unicode_or_bust(obj, encoding='utf-8'):
#     try:
#         if isinstance(obj, basestring):
#             if not isinstance(obj, unicode):
#                 obj = unicode(obj, encoding)
#         else:
#             try:
#                 obj = unicode(obj, encoding)
#             except: # assume iterable and apply to every element
#                 try:
#                     obj = map(lambda x: unicode(x, encoding), obj)
#                 except:
#                     pass
#         return obj
#                 # print "changed something"
#         # print type(obj)
#     except:
#         UnicodeError("to_unicode_or_bust failed with %s" % obj)


def toascii(s):
    '''
    :param s: string or list (of strings)
    :return: string of ascii char correspondents
    (replacing, for example, accentuated letters with non-accentuated versions of the latter)
    '''
    if isinstance(s, str):
        if not isinstance(s, str):  # transform to unicode if it's not already so
            s = str(s, encoding='utf-8')
        return unidecode(s)
    else:  # assume it's an iterable
        if not isinstance(s[0], str):  # transform to unicode if it's not already so (NOTE: Only checked first element
            s = [str(x, encoding='utf-8') for x in s]
        return list(map(unidecode, s))


def lower(s):
    if isinstance(s, str):
        return s.lower()
    else:
        return [x.lower() for x in s]


def strip(s):
    if isinstance(s, str):
        return re.sub(multiple_spaces_exp, ' ', s.strip())
    else:
        return [re.sub(multiple_spaces_exp, ' ', x.strip()) for x in s]


def replace_space_by_underscore(s):
    if isinstance(s, str):
        return s.replace(' ', '_')
    else:
        return [x.replace(' ', '_') for x in s]

    ## with to_unicode_or_bust INSIDE the functions:
    # def toascii(s):
    #     if isinstance(s,basestring):
    #         return unidecode(to_unicode_or_bust(s))
    #     else:
    #         return map(lambda x:unidecode(to_unicode_or_bust(x)),s)
    #
    # def lower(s):
    #     if isinstance(s,basestring):
    #         return to_unicode_or_bust(s).lower()
    #     else:
    #         return map(lambda x:to_unicode_or_bust(x).lower(),s)

    # if isinstance(s,string):
    #     return unidecode(codecs.decode(s, 'utf-8'))
    # elif isinstance(s,unicode):
    #     return unidecode(s)
    # elif isinstance(s,list):
    #     if isinstance(s[0],string):
    #         return map(lambda x:unidecode(codecs.decode(x, 'utf-8')),s)
    #     elif isinstance(s[0],unicode):
    #         return map(lambda x:unidecode(x,s))
