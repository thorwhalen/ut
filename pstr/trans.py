__author__ = 'thorwhalen'

from unidecode import unidecode
import re
import pandas as pd

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
