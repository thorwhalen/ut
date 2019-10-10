__author__ = 'thor'

from numpy import *
import pandas as pd
import itertools

import ut as ms
import ut.daf.ch
import ut.pmath.poset
import ut.pcoll.order_conserving
from ut.util.uiter import powerset

default = dict()
default['success'] = 'success'
default['trial'] = 'trial'
default['rate'] = 'rate'
default['set_elements_name'] = 'element'  # category of the elements used in the sets
default['set_column'] = 'set'  # name of column that contains the sets (which will index the data)


class SetEst(object):
    def __init__(self, d, **kwargs):
        # process inputs
        kwargs = dict(default, **kwargs)
        self.success = kwargs['success']
        self.trial = kwargs['trial']
        self.rate = kwargs['rate']
        self.set_elements_name = kwargs['set_elements_name']
        # get and order data
        self.d = d
        # compute other columns and attributes
        self.index_type = type(self.d.index.values[0])
        self.set_elements = list(unique(list(itertools.chain.from_iterable(self.d.index.values))))
        self.n_elements = len(self.set_elements)
        self.add_bitmap()
        self.add_stats()
        # order columns
        self.d = self.d[self.non_element_columns() + self.set_elements]
        # init poset
        self._poset = None  # will be computes if and when used
        # vector to map bitmaps to ints
        self.hash_base = array([2**i for i in range(self.n_elements)])
        self.hash_base_matrix = matrix(self.hash_base).T
        self.hash_to_value = dict()

    def bitmap_to_hash(self, bitmap):
        return array((bitmap * self.hash_base_matrix))

    # def mk_hash_to_value(self, val_col=None):
    #     val_col = val_col or self.success

    def add_bitmap(self):
        ddd = ms.pmath.poset.family_of_sets_to_bitmap(self.d.index.values)[self.set_elements]
        self.d = pd.concat([self.d, ddd], axis=1)
        # self.d[self.set_elements] = ddd

    def poset(self, num_of_elements=None):
        num_of_elements = num_of_elements or self.n_elements
        if self._poset is None:
            self._poset = ms.pmath.poset.set_containment_matrix(self.bitmap_matrix())
        return self._poset

    def add_stats(self):
        self.d.loc[:, 'n_members'] = list(map(len, self.d.index.values))
        if self.trial in self.d.columns:
            self.d.loc[:, self.rate] = self.d[self.success] / array(list(map(float, self.d[self.trial])))

    def rm_bitmap(self):
        self.d = self.dd()

    def bitmap_matrix(self):
        return self.d[self.set_elements].as_matrix()

    def non_element_columns(self):
        return ms.pcoll.order_conserving.setdiff(list(self.d.columns), self.set_elements)

    def sort_d(self, columns=None, **kwargs):
        self.d = self.d.sort(columns=columns, **kwargs)

    def dd(self):
        return self.d[self.non_element_columns()]

    def get_rate(self, df):
        return df[self.success] / df[self.trial]

    def change_type_of_d_index(self, change_to_type=tuple):
        index_name = self.d.index.name or 'index'
        self.d = self.d.reset_index(drop=False)
        self.d.loc[:, index_name] = self.d.loc[:, index_name].apply(change_to_type)
        self.d = self.d.set_index(index_name)
        self.index_type = type(self.d.index.values[0])  # couldn't I use change_to_type here?

    def subset_summed_d(self, cols=None):
        cols = cols or [self.success, self.trial]
        if isinstance(cols, str):
            cols = [cols]
        t = self.d[cols].copy()
        return pd.DataFrame([sum(t[lidx]) for lidx in self.poset()], index=self.d.index)

    def get_stats_of_subsets_of(self, idx):
        if not isinstance(idx, int):
            idx = type(idx)(unique(idx))
            idx = array([x == idx for x in self.d.index.values])
            t = self.dd()[self.poset()[idx, :].T]
        else:
            t = self.dd().iloc[self.poset()[idx, :]]
        return t.sort(['n_members', self.trial, self.success], ascending=False)


    @staticmethod
    def _process_input_for_factories(df, kwargs):
        kwargs['success'] = kwargs.get('success',
                                       list(set(kwargs.get('set_column', default['set_column']))
                                       .difference(df.columns))[0])
        df = df.copy()
        df[kwargs['set_column']] = list(map(tuple, list(map(unique, df[kwargs['set_column']]))))
        return df, dict(default, **kwargs)

    @staticmethod
    def _mk_data_from_set_success_df(df, **kwargs):
        df = df[[kwargs['set_column'], kwargs['success']]] \
            .groupby(kwargs['set_column']) \
            .agg(['sum', 'count'])[kwargs['success']]
        df = ms.daf.ch.ch_col_names(df, new_names=[kwargs['success'], kwargs['trial']], old_names=['sum', 'count'])
        df = df.sort([kwargs['success'], kwargs['trial']], ascending=False)
        return df


    @staticmethod
    def from_set_success_df(df, **kwargs):
        df, kwargs = SetEst._process_input_for_factories(df, kwargs)
        df = SetEst._mk_data_from_set_success_df(df, **kwargs)
        return SetEst(d=df, **kwargs)

    @staticmethod
    def mk_subset_summed_from_set_success_df(df, **kwargs):
        se = SetEst.from_set_success_df(df, **kwargs)
        se.d[[se.success, se.trial]] = se.subset_summed_d()
        se.add_stats()
        return se

    @staticmethod
    def mk_subset_summed_closure_from_set_success_df(df, **kwargs):
        df, kwargs = SetEst._process_input_for_factories(df, kwargs)
        df = SetEst._mk_data_from_set_success_df(df, **kwargs)
        # make missing combos data and append to existing data
        existing_combos = df.index.values
        set_elements = list(unique(list(itertools.chain.from_iterable(existing_combos))))
        missing_combos = set(powerset(set_elements)).difference(existing_combos)
        missing_combo_data = pd.DataFrame({kwargs['success']: 0, kwargs['trial']: 0}, index=missing_combos)
        # append to existing data
        df = pd.concat([df, missing_combo_data], axis=0)
        # make a SetEst from the set_success df
        se = SetEst(df, **kwargs)
        se.d[[se.success, se.trial]] = se.subset_summed_d()
        se.add_stats()
        return se


class Shapley(SetEst):
    def __init__(self, d, **kwargs):
        super(Shapley, self).__init__(d, **kwargs)
        self.val_col = kwargs.get('val_col', self.success)
        self.subset_val_map = None
        self.compute_subset_val_map()

    def compute_subset_val_map(self):
        self.subset_val_map = {tuple(k): v for k, v in zip(self.bitmap_matrix(), self.d[self.val_col])}

    def get_subset_val(self, subset):
        return self.subset_val_map.get(tuple(subset), 0)

    def compute_shapley_values(self):
        return {element: self._compute_single_shapley_value(element) for element in self.set_elements}

    def _compute_single_shapley_value(self, element):
        t = self._mk_marginal_values_for_element(element)
        tt = t[['subset_sizes', 'success']].groupby('subset_sizes').mean()
        return mean(tt['success'])

    def _mk_marginal_values_for_element(self, element):
        # get location lidx
        element_col_lidx = array([x == element for x in self.set_elements])
        element_row_lidx = array(self.d[element] > 0)
        # sum up all values for subsets containing element
        t = self.d[[self.val_col]][element_row_lidx]
        # get the the values of the subsets obtained by removing this element from the sets it's in
        subsets_intersecting_with_element = self.bitmap_matrix()[element_row_lidx, :]
        t['subset_sizes'] = sum(subsets_intersecting_with_element, axis=1)
        subsets_intersecting_with_element[:, element_col_lidx] = 0
        t['success'] = t[self.val_col] - array(list(map(self.get_subset_val, subsets_intersecting_with_element)))
        return t

    def _compute_single_shapley_value_experimental(self, element):
        def group_stats_fun(g):
            return g['success'].sum() / float(g['subset_sizes'].iloc[0])
        t = self._mk_marginal_values_for_element(element)
        tt = t[['subset_sizes', 'success']].groupby('subset_sizes').apply(group_stats_fun)
        return mean(tt)
        # tt = t[['subset_sizes', 'success']].groupby('subset_sizes').mean()
        # return mean(tt['success'])

    @staticmethod
    def mk_subset_summed_closure_from_set_success_df(df, **kwargs):
        df, kwargs = Shapley._process_input_for_factories(df, kwargs)
        df = Shapley._mk_data_from_set_success_df(df, **kwargs)
        # make missing combos data and append to existing data
        existing_combos = df.index.values
        set_elements = list(unique(list(itertools.chain.from_iterable(existing_combos))))
        missing_combos = set(powerset(set_elements)).difference(existing_combos)
        missing_combo_data = pd.DataFrame({kwargs['success']: 0, kwargs['trial']: 0}, index=missing_combos)
        # append to existing data
        df = pd.concat([df, missing_combo_data], axis=0)
        # make a SetEst from the set_success df
        se = Shapley(df, **kwargs)
        se.d[[se.success, se.trial]] = se.subset_summed_d()
        se.compute_subset_val_map()
        se.add_stats()
        return se


class WithOrWithout(SetEst):
    def __init__(self, d, **kwargs):
        super(WithOrWithout, self).__init__(d, **kwargs)

    def with_and_without_stats_for_element(self, element, extra_group_vars=[]):
        grvars = extra_group_vars + [element]
        dd = self.d[[self.success, self.trial] + grvars].groupby(grvars).sum()
        dd['rate'] = dd[self.success] / dd[self.trial]
        return dd

    def with_and_without_stats(self):
        dd = pd.DataFrame()
        for c in self.set_elements:
            t = self.with_and_without_element_stats(c)
            t['element'] = c
            t['element_present'] = t.index.values
            t = t.reset_index(drop=True)
            dd = pd.concat([dd, t], axis=0)
        return dd.set_index(['element', 'element_present'])

    def with_and_without_rate_lift(self):
        dd = dict()
        for c in self.set_elements:
            t = self.with_and_without_stats_for_element(c)
            dd[c] = t[t.index == 1]['rate'].iloc[0] / t[t.index == 0]['rate'].iloc[0]
        dd = pd.DataFrame({'element': list(dd.keys()), 'rate_lift': list(dd.values())})
        dd = dd.sort('rate_lift', ascending=False).set_index('element')
        return dd