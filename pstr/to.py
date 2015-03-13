__author__ = 'thorwhalen'

import re
import numpy as np

num_re = re.compile('[^\d]+')
format_tags = re.compile('(?<={)[^}]+(?=})')


def formating_args(s):
    return list(set(format_tags.findall(s)))


# def formatted(s, variables):
#     eval(s + '.format(' + ','.join(formating_args(s)) + ')', globals=variables, locals=variables)


def float(x, not_int_val=np.nan):
    return float(x.replace(',', '.'))


def integer(x, not_int_val=np.nan):
    x = num_re.sub('', x)
    if x:
        return int(x)
    else:
        return not_int_val


def file(string, tofile, encoding="UTF-8"):
    if encoding!=None:
        with open(tofile, "wb") as f:
            f.write(string.encode(encoding))
    else:
        text_file = open(tofile, "w")
        text_file.write(string)
        text_file.close()


def _file(string, tofile, encoding="UTF-8"):
    if encoding!=None:
        with open(tofile, "wb") as f:
            f.write(string.encode(encoding))
    else:
        text_file = open(tofile, "w")
        text_file.write(string)
        text_file.close()


def afile(string, tofile, encoding="UTF-8"):
    try:
        if encoding != None:
            with open(tofile, "wb") as f:
                f.write(string.encode(encoding))
        else:
            text_file = open(tofile, "w")
            text_file.write(string)
            text_file.close()
    except UnicodeDecodeError:
        string = string.decode('utf-8')
        if encoding!=None:
            with open(tofile, "wb") as f:
                f.write(string.encode(encoding))
        else:
            text_file = open(tofile, "w")
            text_file.write(string)
            text_file.close()


def no_error_processing_lidx(str_iter, proc_fun):
    lidx = list()
    for s in str_iter:
        try:
            proc_fun(s)
            lidx.append(True)
        except Exception:
            lidx.append(False)
    return lidx


def convert_camelcase_to_lower_underscore(string):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

