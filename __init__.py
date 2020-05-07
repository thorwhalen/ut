__author__ = 'thorwhalen'

from warnings import warn

from collections import Counter, defaultdict
from datetime import datetime
from pprint import PrettyPrinter
import json
import matplotlib.pylab as plt

ddir = lambda o: [a for a in dir(o) if not a.startswith('_')]
dddir = lambda o: [a for a in dir(o) if not a.startswith('__')]


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


with ModuleNotFoundIgnore():
    from py2store import ihead, kvhead, QuickStore, LocalBinaryStore, LocalJsonStore, LocalPickleStore, LocalTextStore
    from py2store import kv_wrap, wrap_kvs, filtered_iter, cached_keys
    from py2store.util import lazyprop
    from py2store.my.grabbers import grabber_for as _grabber_for

    igrab = _grabber_for('ipython')

with ModuleNotFoundIgnore():
    from py2mint.doc_mint import doctest_string_print, doctest_string

with ModuleNotFoundIgnore():
    import ut.daf.ch
    import ut.daf.manip
    import ut.daf.gr
    import ut.daf.to
    from ut.daf.diagnosis import diag_df as diag_df

with ModuleNotFoundIgnore():
    import ut.util.pstore

with ModuleNotFoundIgnore():
    from ut.pcoll.num import numof_trues

with ModuleNotFoundIgnore():
    from ut.util.log import printProgress, print_progress

with ModuleNotFoundIgnore():
    import ut.pplot.distrib

with ModuleNotFoundIgnore():
    from ut.pplot.matrix import heatmap
    from ut.pplot.my import vlines

with ModuleNotFoundIgnore():
    import numpy as np

with ModuleNotFoundIgnore():
    import pandas as pd

with ModuleNotFoundIgnore():
    from ut.sh.py import add_to_pythonpath_if_not_there
