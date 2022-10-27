from collections import defaultdict
import numpy as np


class MatrixDataFrame(object):
    def __init__(self, data=None, index=None, columns=None):
        if data is None:
            data = {}

        if isinstance(data, MatrixDataFrame):
            data = data._d

        if isinstance(data, dict):
            self._d = np.hstack(list(data.values()))
        elif isinstance(data, (np.ndarray, list, np.matrix)):
            n = len(data)
            if n == 0:
                self._d = np.array([])
            else:
                if isinstance(data[0], dict):
                    col_data = defaultdict(list)
                    for d in data:
                        for k, v in d.items():
                            col_data[k].append(v)
                    self._d = np.array(col_data)
                else:
                    self._d = np.array(data)

        if index is None:
            index = np.arange(self._d.shape[0])
        if columns is None:
            columns = np.arange(self._d.shape[1])

        self.index = np.array(index)
        self.columns = np.array(columns)

    def sum(self, axis=0):
        if axis == 0:
            data = self._d.sum(axis=axis)
            return MatrixDataFrame(data=data, columns=self.columns)
        elif axis == 1:
            raise NotImplementedError('not implemented yet')

    def __str__(self):
        s = ''
        s += '{}\n'.format(self.columns)
        s += '{}'.format(self._d)
        return s

    def __repr__(self):
        return self.__str__()


class RecDataFrame(object):
    def __init__(self, data=None, index=None, columns=None):
        if isinstance(data, np.recarray):
            self._d = data
        elif isinstance(data, (list, np.ndarray, np.matrix)):
            if columns is not None:
                self._d = np.rec.fromarrays(data, names=columns)
            else:
                self._d = np.rec.fromarrays(data)
        else:
            raise TypeError(
                "Don't know how to construct a RecDataFrame with that input type"
            )

        if index is None:
            index = np.arange(self._d.shape[0])
        self.index = index

    @property
    def columns(self):
        return np.array(self._d.dtype.names)

    def __str__(self):
        return self._d.__str__()

    def __repr__(self):
        return self._d.__repr__()


class DictDataFrame(object):
    def __init__(self, data=None, index=None, columns=None):
        pass
