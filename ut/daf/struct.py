__author__ = 'thor'

import ut as ms
import pandas as pd
import ut.pcoll.order_conserving
from functools import reduce


class SquareMatrix:
    def __init__(self, df, index_vars=None, sort=False):
        if isinstance(df, SquareMatrix):
            self = df.copy()
        elif isinstance(df, pd.DataFrame):
            self.df = df
            self.index_vars = index_vars
            self.value_vars = ms.pcoll.order_conserving.setdiff(
                list(self.df.columns), self.index_vars
            )
            self.df = self.df[self.index_vars + self.value_vars]
        else:
            raise NotImplementedError("This case hasn't been implemented yet")

        if sort:
            self.df.sort(columns=self.index_vars, inplace=True)

    def copy(self):
        return SquareMatrix(df=self.df.copy(), index_vars=self.index_vars)

    def transpose(self):
        return SquareMatrix(
            df=self.df, index_vars=[self.index_vars[1], self.index_vars[0]]
        )

    def reflexive_mapreduce(self, map_fun, reduce_fun=None, broadcast_functions=True):
        df = self.df.merge(
            self.df,
            how='inner',
            left_on=self.index_vars[1],
            right_on=self.index_vars[0],
            suffixes=('', '_y'),
        )

        df[self.index_vars[1]] = df[self.index_vars[1] + '_y']
        df.drop(
            labels=[self.index_vars[0] + '_y', self.index_vars[1] + '_y'],
            axis=1,
            inplace=True,
        )

        if not isinstance(map_fun, dict) and broadcast_functions:
            map_fun = dict(list(zip(self.value_vars, [map_fun] * len(self.value_vars))))
        for k, v in map_fun.items():
            df[k] = v(df[k], df[k + '_y'])
        df.drop(labels=[x + '_y' for x in self.value_vars], axis=1, inplace=True)

        if not reduce_fun:
            reduce_fun = dict()
            for k, v in map_fun.items():
                reduce_fun[k] = lambda x: reduce(v, x)
        elif not isinstance(reduce_fun, dict) and broadcast_functions:
            reduce_fun = dict(
                list(zip(self.value_vars, [reduce_fun] * len(self.value_vars)))
            )
        df = df.groupby(self.index_vars).agg(reduce_fun).reset_index(drop=False)

        return SquareMatrix(df=df, index_vars=self.index_vars)

    def reverse_indices(self):
        return [self.index_vars[1], self.index_vars[0]]

    def sort(self, **kwargs):
        kwargs = dict({'columns': self.index_vars}, **kwargs)
        sm = self.copy()
        sm.df = sm.df.sort(**kwargs)
        return sm

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.df.set_index(self.index_vars).__str__()

    def head(self, num_of_rows=5):
        return self.df.head(num_of_rows)

    def tail(self, num_of_rows=5):
        return self.df.tail(num_of_rows)
