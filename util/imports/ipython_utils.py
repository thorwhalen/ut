__author__ = 'thor'

print('''
Remember to use these useful utils:
    igrab, QuickStore, set_obj, get_obj, doctest_string_print
    heatmap, vlines
    print_progress, ppr, pickle_dump, pickle_load, numof_trues
''')

import sys

if sys.platform == 'darwin':
    from IPython.display import set_matplotlib_formats

    set_matplotlib_formats('retina')

from numpy import *
from py2store import set_obj, get_obj

from py2store import QuickStore
from py2store.my.grabbers import grabber_for as _grabber_for

igrab = _grabber_for('ipython')

from py2mint.doc_mint import doctest_string_print, doctest_string

import os
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

import ut.daf.ch
import ut.daf.manip
import ut.daf.gr
import ut.daf.to
import ut.util.pstore
# import ut.pfile.accessor

from datetime import datetime

from ut.pcoll.num import numof_trues
# from ut.util.pstore import MyStore
from ut.util.log import printProgress, print_progress

from ut.daf.diagnosis import diag_df as diag_df

import ut.pplot.distrib

from pprint import PrettyPrinter
import json
from ut.util.pstore import pickle_dump, pickle_load

from ut.pplot.matrix import heatmap
from ut.pplot.my import vlines


class PPR(object):
    indent = 2
    pretty_printer = PrettyPrinter(indent=indent)

    @classmethod
    def __call__(cls, x):
        if isinstance(x, dict):
            cls.pdict(x)
        else:
            cls.pprint(x)

    @classmethod
    def pprint(cls, x):
        cls.pretty_printer.pprint(x)

    @classmethod
    def pjson(cls, x):
        print(json.dumps(x, indent=cls.indent))

    @classmethod
    def pdict(cls, x):
        print(cls.format_str(x))

    @classmethod
    def format_str(cls, x, tab=0):
        s = ['{\n']
        for k, v in list(x.items()):
            if isinstance(v, dict):
                v = cls.format_str(v, tab + cls.indent)
            else:
                v = repr(v)

            s.append('%s%r: %s,\n' % ('  ' * tab, k, v))
        s.append('%s}' % ('  ' * tab))
        return ''.join(s)


def ppr(x):
    PPR.__call__(x)


def see_linked_header(text, level=0, link_to_sections=None, indent_size=3):
    if link_to_sections is None:
        if level <= 3:
            link_to_sections = True
        else:
            link_to_sections = False

    section_level_bullet = [
        "&#8227; ",
        "&#8250; ",
        "&#8226; ",
        "&#8208; ",
        "&#8901; ",
        " "
    ]

    text = text.replace('"', "'")
    single_indent = "&nbsp;" * indent_size
    indent = single_indent * level
    bullet = section_level_bullet[min(level, len(section_level_bullet))]
    header_prefix = "#" + "#" * level

    section = '{indent}{bullet}<a href="#{text}">{text}</a><br>'.format(indent=indent, bullet=bullet, text=text)
    header = '<p><a name="{text}"></a></p>\n{header_prefix} {text}'.format(header_prefix=header_prefix, text=text)

    if link_to_sections:
        header += ' [^](#sections) '

    print(section)
    print("")
    print(header)
