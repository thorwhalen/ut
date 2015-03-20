__author__ = 'thor'


print '''
    printProgress
    numof_trues
    ppr
'''

import os
import re
import pandas as pd
import numpy as np
from numpy import *

import ut.daf.ch
import ut.daf.manip
import ut.daf.gr
import ut.daf.to
import ut.util.pstore
import ut.pfile.accessor

from datetime import datetime

from ut.pcoll.num import numof_trues
# from ut.util.pstore import MyStore
from ut.util.log import printProgress

from ut.daf.diagnosis import diag_df as diag_df

import ut.pplot.distrib

from pprint import PrettyPrinter
import json


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
        print json.dumps(x, indent=cls.indent)

    @classmethod
    def pdict(cls, x):
        print cls.format_str(x)

    @classmethod
    def format_str(cls, x, tab=0):
        s = ['{\n']
        for k, v in x.items():
            if isinstance(v, dict):
                v = cls.format_str(v, tab+cls.indent)
            else:
                v = repr(v)

            s.append('%s%r: %s,\n' % ('  '*tab, k, v))
        s.append('%s}' % ('  '*tab))
        return ''.join(s)


def ppr(x):
    PPR.__call__(x)