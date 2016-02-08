__author__ = 'thor'

import pandas as pd
from numpy import *
from matplotlib.pyplot import *
import numpy as np
from collections import Counter
# from numpy.random import rand
# from numpy.random import permutation

import ut.pcoll.order_conserving as colloc

# import ut as ut
from ut.daf.op import cartesian_product
from ut.daf.gr import group_and_count
from ut.daf.ch import ch_col_names
from ut.daf.manip import reorder_columns_as
from ut.util.ulist import ascertain_list
from ut.util.prand import rand_numbers_summing_to_one
from ut.pplot.color import shifted_color_map

import ut.pplot.get


class Pot(object):
    def __init__(self, data=None):
        if isinstance(data, Pot):
            self.tb = data.tb
        elif isinstance(data, float) or isinstance(data, int):
            self.tb = pd.DataFrame([{'pval': data}])
        elif data is not None:
            if isinstance(data, pd.DataFrame):
                # inject the dataframe in the tb attribute: It's the potential data
                assert 'pval' in data.columns, "dataframe had no pval column"
                self.tb = data
            elif isinstance(data, dict):
                if 'pval' not in data.keys():
                    data = dict(data, pval=len(data[data.keys()[0]]) * [1])
                self.tb = pd.DataFrame(data=data)
            else:
                try:
                    self.tb = data.tb.copy()
                except Exception:
                    raise ValueError("Unknown construction type")
        else:
            self.tb = pd.DataFrame({'pval': 1}, index=[''])  # default "unit" potential
        self.tb.index = [''] * len(self.tb)

    def vars(self):
        return colloc.setdiff(list(self.tb.columns), ['pval'])

    ###########################################
    # OPERATIONS
    ###########################################
    def get_slice(self, intercept_dict):
        """
        Return sub-pot going through specific "intercept points"
        For example, if X is a pot on ABC, then X.get_slice({'A':0, 'B':1}) is the pot on C taken from ABC where
        A=0 and B=1.
        It's like a subplane of points defined by given axis intercepts.
        """
        tb = self.tb.copy()
        for k, v in intercept_dict.iteritems():
            tb = tb[tb[k] == v]
            del tb[k]
        return Pot(tb)

    def project_to(self, var_list=[]):
        """
        project to a subset of variables (marginalize out other variables)
        """
        var_list = colloc.intersect(ascertain_list(var_list), self.vars())
        if var_list:  # if non-empty, marginalize out other variables
            return Pot(self.tb[var_list + ['pval']].groupby(var_list).sum().reset_index())
        else:  # if _var_list is empty, return a singleton potential containing the sum of the vals of self.tb
            return Pot(pd.DataFrame({'pval': self.tb['pval'].sum()}, index=['']))

    def __rshift__(self, var_list):
        return self.project_to(var_list)

    def normalize(self, var_list=[]):
        """
        'Normalization' of the pot with respect to _var_list.
        Will define the pot by the projection of the pot on a subset of the variables.

        Note: If this subset is the empty set, this will correspond to "full normalization", i.e. dividing the vals by
        the sum of all vals.

        Use:
            * This can be used to transform a count potential into a probability potential
            (if your sample is large enough!)
            * Conditional Probability: P(A|B) = P(AB) / P(B)
        """
        return self / self.project_to(var_list)

    def __or__(self, item):
        """
        If item is empty/none/false, a string or a list, it normalizes according to item.
        If item is a dict, it normalizes according to the keys, and slices according to the dict.
        --> This resembles P(A|B=1) kind of thing...
        """
        print "I'm trying to discourage using | now (might want to use it for fuzzy logic at some point"
        print "--> Use / instead of |. "
        if isinstance(item, basestring):
            return self / self.project_to([item])
        elif isinstance(item, list):
            return self / self.project_to(item)
        elif isinstance(item, dict):
            intercept_dict = item
            var_list = colloc.intersect(self.vars(), intercept_dict.keys())
            return (self / self.project_to(var_list)).get_slice(intercept_dict)
        else:
            TypeError('Unknown item type')

    def __getitem__(self, item):
        """
        This function is called when accessing the pot with [] brackets, and will return a slice of projection of the
        pot depending on the type of item.
        """
        if item:
            if isinstance(item, dict):
                return self.get_slice(item)
            elif isinstance(item, list):
                return self.project_to(item)
            elif isinstance(item, basestring):
                return self.project_to(item)
            else:
                raise TypeError("Unknown type for item (must be None, dict, list, or string)")
        else:
            return Pot(pd.DataFrame({'pval': self.tb['pval'].sum()}, index=['']))

    def __add__(self, pot):
        return Pot(_val_add_(self._merge_(pot)))
        # if isinstance(y, float) | isinstance(y, int):
        #     self.tb['pval'] += y
        # else:
        #     pass

    def __mul__(self, pot):
        """
        Multiply two potentials
        """
        return Pot(_val_prod_(self._merge_(pot)))

    def __div__(self, item):
        """
        Operation depends on what item's type is. If item is a:
            Pot: perform potential division (like multiplication but with pvals divided).
            empty/none/false, a string or a list: normalize according to item.
            dict: it normalizes according to the keys, and slices according to the dict.
        --> This resembles P(A|B=1) kind of thing...
        """
        if isinstance(item, Pot):
            return Pot(_val_div_(self._merge_(item)))
        elif isinstance(item, basestring):
            return self.normalize([item])
        elif isinstance(item, list):
            return self.normalize(item)
        elif isinstance(item, dict):
            intercept_dict = item
            var_list = colloc.intersect(self.vars(), intercept_dict.keys())
            return self.normalize(var_list).get_slice(intercept_dict)
        else:
            TypeError('Unknown item type')

    def __truediv__(self, item):
        return self.__div__(item)

    def assimilate(self, pot):
        """
        Assimilate information given by input pot (returning the result).
        Assimilation means multiplication followed by a projection to the original variables.
        This is used, for example, when wanting to compute P(X|D=data) as the normalization of P(D=data|X) * P(X)
        (Bayes rule). We can write that as P(X) absorbing P(D=data|X). The result has the dimensions of X.
        """
        return self.__mul__(pot).normalize([]).project_to(self.vars())

    def unassimilate(self, pot):
        """
        Inverse of assimilate.
        """
        return self.__div__(pot).normalize([]).project_to(self.vars())

    ###########################################
    # Usable UTILS
    ###########################################
    def order_vars(self, var_list, sort_pts=True):
        self.tb = reorder_columns_as(self.tb, ascertain_list(var_list))
        if sort_pts:
            self.sort_pts()
        return self

    def sort_pts(self, var_list=None, **kwargs):
        var_list = var_list or self.vars()
        self.tb = self.tb.sort_values(by=var_list, **kwargs)
        return self

    def pval(self):
        return self.tb.pval

    def pval_of(self, var_val_dict, default_val=0.0):
        t = self.get_slice(var_val_dict)
        n = len(t.tb)
        if n == 0:
            return default_val
        elif n == 1:
            return t.tb.pval[0]
        else:
            raise RuntimeError("In pval_of(): get_slice returned more than one value")

    def binarize(self, var_values_to_map_to_1_dict):
        """
        maps specified variables to {0, 1}
            var_values_to_map_to_1_dict is a {variable_name: values to map to 1} specification dict
        """
        for var_name, vals_to_map_to_1 in var_values_to_map_to_1_dict.iteritems():
            tb = self.tb.copy()
            if not hasattr(vals_to_map_to_1, '__iter__'):
                vals_to_map_to_1 = [vals_to_map_to_1]
            lidx = tb[var_name].isin(vals_to_map_to_1)
            tb[var_name] = 0
            tb[var_name].loc[lidx] = 1
        tb = tb.groupby(self.vars()).sum().reset_index(drop=False)
        return Pot(tb)

    def round(self, ndigits=None, inplace=False):
        if ndigits is None:
            ndigits = abs(int(math.log10(self.tb['pval'].min()))) + 1 + 2
            print ndigits
        rounded_pvals = map(lambda x: round(x, ndigits), self.tb['pval'])
        if inplace:
            self.tb['pval'] = rounded_pvals
        else:
            x = Pot(self)
            x.tb['pval'] = rounded_pvals
            return x

    def rect_perspective_df(self):
        vars = self.vars()
        assert len(self.vars()) == 2, "You can only get the rect_perspective_df of a pot with exactly two variables"
        return self.tb.set_index([vars[0], vars[1]]).unstack(vars[1])['pval']

    ###########################################
    # Hidden UTILS
    ###########################################
    def _merge_(self, pot):
        """
        Util function. Shouldn't really be used directly by the user.
        Merge (join) two pots.
        An inner merge of the two pots, on the intersection of their variables (if non-empty) will be performed,
        producing val_x and val_y columns that will contain the original left and right values, aligned with the join.
        Note: If the vars intersection is empty, the join will correspond to the cartesian product of the variables.
        """
        on = colloc.intersect(self.vars(), pot.vars())  # we will merge on the intersection of the variables (not pval)
        if on:
            return pd.merge(self.tb, pot.tb, how='inner', on=on, sort=True, suffixes=('_x', '_y'))
        else:  # if no common variables, take the cartesian product
            return cartesian_product(self.tb, pot.tb)

    def __str__(self):
        """
        This will return a string that represents the underlying dataframe (used when printing the pot)
        """
        return self.tb.__repr__()

    def __repr__(self):
        """
        This is used by iPython to display a variable.
        I chose to do thing differently than __str__.
        Here the dataframe is indexed by the vars and then made into a string.
        This provides a hierarchical progression perspective to the variable combinations.
        """
        if self.vars():
            return self.tb.set_index(self.vars()).__str__()
        else:
            return self.tb.__repr__()

    #def assert_pot_validity(self):
    #    assert 'pval' in self.tb.columns, "the potential dataframe has no column named 'pval'"
    #    assert len(self.tb.)


    #################################################################################
    # FACTORIES

    @classmethod
    def binary_pot(cls, varname, prob=1):
        return Pot(pd.DataFrame({varname: [0, 1], 'pval': [1 - prob, prob]}))

    @classmethod
    def from_points_to_count(cls, pts, vars=None):
        """
        By "points" I mean a collection (through some data structure) of multi-dimensional coordinates.
        By default, all unique points will be grouped and the pval will be the cardinality of each group.
        """
        if isinstance(pts, pd.DataFrame):
            # tb = group_and_count(pts)
            # tb = ch_col_names(tb, 'pval', 'count')
            return Pot(group_and_count(pts, count_col='pval'))
        else:
            counts = Counter(pts)
            if vars is None:
                example_key = counts.keys()[0]
                vars = range(len(example_key))
            return Pot(pd.DataFrame(
                [dict(pval=v, **{kk: vv for kk, vv in zip(vars, k)}) for k, v in counts.iteritems()])
            )


    @classmethod
    def from_count_df_to_count(cls, count_df, count_col='pval'):
        """
        Creates a potential from a dataframe specifying point counts (where the count column name is specified by
        count_col
        """
        pot_vars = list(colloc.setdiff(count_df.columns, [count_col]))
        tb = count_df[pot_vars+[count_col]].groupby(pot_vars).sum().reset_index()
        tb = ch_col_names(tb, 'pval', count_col)
        return Pot(tb)

    @classmethod
    def from_points_to_bins(cls, pts, **kwargs):
        """
        Creates a potential from a dataframe specifying point counts (where the count column name is specified by
        count_col
        """
        if isinstance(pts, pd.DataFrame):

            tb = group_and_count(pts)
            tb = ch_col_names(tb, 'pval', 'count')
            return Pot(tb)

    @classmethod
    def rand(cls, n_var_vals=[2, 2], var_names=None, granularity=None, try_to_get_unique_values=False):
        # check inputs
        assert len(n_var_vals) <= 26, "You can't request more than 26 variables: That's just crazy"
        if var_names is None:
            var_names = [str(unichr(x)) for x in range(ord('A'),ord('Z'))]
        assert len(n_var_vals) <= len(var_names), "You can't have less var_names than you have n_var_vals"
        assert min(array(n_var_vals)) >= 2, "n_var_vals elements should be >= 2"

        # make the df by taking the cartesian product of the n_var_vals defined ranges
        df = reduce(cartesian_product, [pd.DataFrame(data=range(x), columns=[y]) for x, y in zip(n_var_vals, var_names)])

        n_vals = len(df)

        def _get_random_pvals():
            if granularity is None:
                if n_vals > 18:
                    x = np.random.rand(n_vals)
                    return x / sum(x)
                elif n_vals == 4:
                    return np.random.permutation([0.1, 0.2, 0.3, 0.4])
                else:
                    if n_vals <= 12:
                        return rand_numbers_summing_to_one(n_vals, 0.05)
                    else:
                        return rand_numbers_summing_to_one(n_vals, 0.01)
            else:
                return rand_numbers_summing_to_one(n_vals, granularity)

        # choose random vals
        if try_to_get_unique_values:
            if not isinstance(try_to_get_unique_values, int):
                try_to_get_unique_values = 1000
            for i in range(try_to_get_unique_values):
                pvals = _get_random_pvals()
                if len(unique(pvals)) == n_vals:
                    break
        else:
            pvals = _get_random_pvals()

        df['pval'] = map(float, pvals)

        return Pot(df)


class ProbPot(Pot):
    def __init__(self, data=None):
        super(ProbPot, self).__init__(data=data)

    def prob_of(self, var_val_dict):
        t = self.get_slice(var_val_dict)
        n = len(t.tb)
        if n == 0:
            return 0.0
        elif n == 1:
            return t.tb.pval[0]
        else:
            raise RuntimeError("In prob_of(): get_slice returned more than one value")

    def given(self, conditional_vars):
        return ProbPot(self.__div__(conditional_vars))

    def relative_risk(self, event_var, exposure_var, event_val=1, exposed_val=1):
        prob = self >> [event_var, exposure_var]
        prob.binarize({event_var: event_val, exposure_var: exposed_val})
        return (prob / {exposure_var: 1})[{event_var: 1}] \
               / (prob / {exposure_var: 0})[{event_var: 1}]

    @staticmethod
    def plot_relrisk_matrix(relrisk):
        t = relrisk.copy()
        matrix_shape = (t['exposure'].nunique(), t['event'].nunique())
        m = ut.daf.to.map_vals_to_ints_inplace(t, cols_to_map=['exposure'])
        m = m['exposure']
        ut.daf.to.map_vals_to_ints_inplace(t, cols_to_map={'event': dict(zip(m, range(len(m))))})
        RR = zeros(matrix_shape)
        RR[t['exposure'], t['event']] = t['relative_risk']
        RR[range(len(m)), range(len(m))] = nan

        RRL = np.log2(RR)
        def normalizor(X):
            min_x = nanmin(X)
            range_x = nanmax(X) - min_x
            return lambda x: (x - min_x) / range_x
        normalize_this = normalizor(RRL)
        center = normalize_this(0)



        color_map = shifted_color_map(cmap=cm.get_cmap('coolwarm'), start=0, midpoint=center, stop=1)
        imshow(RRL, cmap=color_map, interpolation='none');

        xticks(range(shape(RRL)[0]), m, rotation=90)
        yticks(range(shape(RRL)[1]), m)
        cbar = colorbar()
        cbar.ax.set_yticklabels(["%.02f" % x for x in np.exp2(array(ut.pplot.get.get_colorbar_tick_labels_as_floats(cbar)))])

#
#
# class ValPot(Pot):
#     def __init__(self, **kwargs):
#         super(ValPot, self).__init__(**kwargs)

##### Data Prep utils
def from_points_to_binary(d, mid_fun=median):
    dd = d.copy()
    columns = d.columns
    for c in columns:
        dd[c] = map(int, d[c] > mid_fun(d[c]))
    return dd


##### Other utils

def relative_risk(joint_prob_pot, event_var, exposure_var):
    prob = joint_prob_pot >> [event_var, exposure_var]
    return (prob / {exposure_var: 1})[{event_var: 1}] \
           / (prob / {exposure_var: 0})[{event_var: 1}]


def _val_prod_(tb):
    """
    multiplies column val_x and val_y creating column pval (and removing val_x and val_y)
    """
    tb['pval'] = tb['pval_x'] * tb['pval_y']
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _val_div_(tb):
    """
    divides column val_x and val_y creating column pval (and removing val_x and val_y)
    Note: 0/0 will be equal to 0
    """
    tb['pval'] = np.true_divide(tb['pval_x'], tb['pval_y']).fillna(0)
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb


def _val_add_(tb):
    """
    divides column val_x and val_y creating column pval (and removing val_x and val_y)
    Note: 0/0 will be equal to 0
    """
    tb['pval'] = tb['pval_x'] + tb['pval_y']
    tb.drop(labels=['pval_x', 'pval_y'], axis=1, inplace=True)
    return tb







