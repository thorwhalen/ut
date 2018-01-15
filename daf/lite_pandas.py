from __future__ import division

import types
from collections import defaultdict
import pandas as pd
import numpy as np

pd.DataFrame

class DataFrame(object):
    def __init__(self, data=None, index=None, columns=None):
        if data is None:
            data = {}

        if isinstance(data, DataFrame):
            data = data._d

        if isinstance(data, dict):
            self._d = np.hstack(data.values())
        elif isinstance(data, (np.ndarray, list)):
            n = len(data)
            if n == 0:
                self._d = np.array([])
            else:
                if isinstance(data[0], dict):
                    col_data = defaultdict(list)
                    for d in data:
                        for k, v in d.iteritems():
                            col_data[k].append(v)
                    self._d = np.array(col_data)

        if index is None:
            index = np.arange(self._d.shape[0])
        if columns is None:
            columns = np.arange(self._d.shape[1])

        self.index = np.array(index)
        self.columns = np.array(columns)

