__author__ = 'thorwhalen'

from warnings import warn

from collections import Counter, defaultdict
from datetime import datetime
from pprint import PrettyPrinter
import json
from contextlib import suppress as _suppress

# class ModuleNotFoundIgnore:
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if exc_type is ModuleNotFoundError:
#             pass
#         return True


# module_not_found_ignore = ModuleNotFoundIgnore()
ModuleNotFoundIgnore = lambda: _suppress  # temporary, for back-compatibility
module_not_found_ignore = _suppress(ModuleNotFoundError)

try:
    import matplotlib.pylab as plt
except ModuleNotFoundError as e:
    warn(f"{e}: {e.args}")

ddir = lambda o: [a for a in dir(o) if not a.startswith('_')]
dddir = lambda o: [a for a in dir(o) if not a.startswith('__')]


with module_not_found_ignore:
    from i2.deco import preprocess, postprocess, preprocess_arguments, input_output_decorator
    from i2.deco import wrap_class_methods_input_and_output
    from i2.signatures import Sig

with module_not_found_ignore:
    from ut.util.my_proj_populate import populate_proj_from_url

with module_not_found_ignore:
    from py2store import ihead, kvhead, QuickStore, LocalBinaryStore, LocalJsonStore, LocalPickleStore, LocalTextStore
    from py2store import kv_wrap, wrap_kvs, filt_iter, cached_keys
    from py2store.util import lazyprop
    from py2store.my.grabbers import grabber_for as _grabber_for

    igrab = _grabber_for('ipython')

with module_not_found_ignore:
    from i2.doc_mint import doctest_string_print, doctest_string

with module_not_found_ignore:
    from ut.util.context_managers import TimerAndFeedback

with module_not_found_ignore:
    import ut.daf.ch
    import ut.daf.manip
    import ut.daf.gr
    import ut.daf.to
    from ut.daf.diagnosis import diag_df as diag_df

with module_not_found_ignore:
    import ut.pdict.get

with module_not_found_ignore:
    import ut.util.pstore

with module_not_found_ignore:
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer, KNeighborsRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer

with module_not_found_ignore:
    from ut.pcoll.num import numof_trues

with module_not_found_ignore:
    from ut.util.log import printProgress, print_progress

with module_not_found_ignore:
    import ut.pplot.distrib

with module_not_found_ignore:
    from ut.pplot.matrix import heatmap
    from ut.pplot.my import vlines

with module_not_found_ignore:
    import numpy as np

with module_not_found_ignore:
    import pandas as pd

with module_not_found_ignore:
    from ut.sh.py import add_to_pythonpath_if_not_there

with module_not_found_ignore:
    from ut.util.importing import import_from_dot_string

with module_not_found_ignore:
    from ut.net.viz import dgdisp, dagdisp, horizontal_dgdisp

with module_not_found_ignore:
    from ut.util.ipython import all_table_of_contents_html_from_notebooks

with module_not_found_ignore:
    from grub import CodeSearcher


    def grub_code(query, module):
        search = CodeSearcher(module).fit()
        return search(query)
