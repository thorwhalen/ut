__author__ = 'thor'

from collections import Counter, defaultdict
from itertools import permutations
import pandas as pd
from pandas import Series, DataFrame
from numpy import array
from scipy.stats import chi2_contingency

from ut.ppi.pot import Pot
from ut.pdict.special import DictDefaultDict


class EdgeCounter(object):
    def __init__(self, item_set_iterator=None):
        self.num_of_sets = 0
        self.node = Counter()
        self.edge = Counter()
        if item_set_iterator is not None:
            for item_set in item_set_iterator:
                self.update(item_set)

    def update(self, item_set):
        self.num_of_sets += 1
        self.node.update(item_set)
        self.edge.update(permutations(item_set, 2))

    def __repr__(self):
        s = ''
        s += 'num of sets: %d\n' % self.num_of_sets
        s += 'num of nodes: %d\n' % len(self.node)
        s += 'sum of node counts: %d\n' % sum([v for v in self.node.values()])
        s += 'num of edges: %d\n' % len(self.edge)
        s += 'sum of edge counts: %d\n' % sum([v for v in self.edge.values()])
        return s

    def contingency_table(self, var, var2=None):
        """
        Returns a (numpy 2x2 array) contingency tables of the counts of
            [[~var & ~var2, ~var & var2],
             [var & ~var2, var & var2]]
        """
        if var2 is None:
            var2 = var[1]
            var = var[0]
        n11 = self.edge[(var, var2)]
        n_1 = self.node[var2]
        n1_ = self.node[var]
        return array(
            [[self.num_of_sets - n_1 - n1_ + n11, n_1 - n11], [n1_ - n11, n11]]
        )

    def contingency_table_stats(self, contingency_table_stats_fun='chi2_pvalue'):
        """
        returns a dict with the same size and whose (var, var2) keys are those of as self.edge,
        and whose values are contingency_table_stats_fun(contingency_table(var, var2))

        contingency_table_stats_fun defaults to scipy.stats.chi2_contingency(contingency_table)[1]
        (the p-value of the chi square test.
        """
        if isinstance(contingency_table_stats_fun, str):
            if contingency_table_stats_fun == 'chi2_pvalue':
                contingency_table_stats_fun = lambda x: chi2_contingency(x)[1]
            else:
                ValueError('Unknown contingency_table_stats_fun')

        return {
            (var, var2): contingency_table_stats_fun(self.contingency_table(var, var2))
            for var, var2 in self.edge.keys()
        }


class NaiveGraph(object):
    """
    A naive graph is essentially a collection of Naive Bayes Networks, when propagation_depth is 1 (the default).
    """

    def __init__(self, data=None, propagation_depth=1, **kwargs):

        self.propagation_depth = propagation_depth

    def assimilate_evidence(self, evidence):
        if isinstance(evidence, str):
            evidence = Pot.binary_pot(varname=evidence, prob=0.999999)
        evidence_var = evidence.vars()[0]
        self.post_pot[evidence_var] = self.post_pot[evidence_var].assimilate(evidence)
        for adj_var, adj_pot in self.edge[evidence_var].items():
            # print "+++ %s: %.09f" % (adj_var, self.post_pot[adj_var].tb.pval[1])
            self.post_pot[adj_var] = self.post_pot[adj_var].assimilate(
                adj_pot.__mul__(evidence)
            )

    def unassimilate_evidence(self, evidence):
        """
        This is SUPPPOSED to be the inverse of assimilation.

        Assimilating and unassimilating evidence (by multiplying, then dividing, a potential,
        which is what the methods assimilate_evidence() and unassimilate_evidence() do),
        you don't always fall back on your feet very well.

        The closer the evidence is to hard, the worse it gets.
        I have now made the default "hard" evidence be 0.9999 probability instead of 1.0000,
        to take care of this problem somewhat.

        But in practice, it may be better to implement this unassimilation otherwise
        (by resetting everything, and reintroducing the remaining evidence
        (minus the one unassimilated) one by one).
        """
        if isinstance(evidence, str):
            evidence = Pot.binary_pot(varname=evidence, prob=0.999999)
        evidence_var = evidence.vars()[0]
        self.post_pot[evidence_var] = self.post_pot[evidence_var].unassimilate(evidence)
        for adj_var, adj_pot in self.edge[evidence_var].items():
            self.post_pot[adj_var] = (
                self.post_pot[adj_var].__div__(
                    adj_pot.__mul__(evidence).project_to(adj_var)
                )
            ).normalize()

    def __repr__(self):
        s = ''
        s += 'num of nodes: %d\n' % len(self.node)
        s += 'num of (directed) edges: %d\n' % sum([len(v) for v in self.edge.values()])
        return s


class BinaryNaiveGraph(NaiveGraph):
    """
    A naive graph is essentially a collection of Naive Bayes Networks, when propagation_depth is 1 (the default).
    """

    def __init__(
        self,
        data=None,
        propagation_depth=1,
        prior_denominator=0.0,
        prior_numerator=0.0,
        **kwargs
    ):
        super(BinaryNaiveGraph, self).__init__(
            data=data, propagation_depth=propagation_depth
        )

        if hasattr(data, '__iter__'):
            data = EdgeCounter(data)

        if isinstance(data, EdgeCounter):
            n = data.num_of_sets
            a = Series(data.node)
            a.index.names = ['A']
            ba_count = Series(data.edge)
            ba_count.index.names = ['B', 'A']

            self.prior_denominator = float(prior_denominator)
            prior_item_prob = kwargs.get(
                'prior_item_prob', len(a) / float(sum(ba_count))
            )
            self.prior_numerator = float(
                prior_numerator or prior_denominator * prior_item_prob
            )
            assert (
                self.prior_numerator <= self.prior_denominator
            ), 'Violation of prior_numerator <= prior_denominator'

            if self.prior_denominator != 0:
                prior_item_prob = self.prior_numerator / self.prior_denominator
                # the prior (mean) number of pairs of items in a same set, assuming item probs are independent
                prior_item_item_numerator = (
                    prior_item_prob * prior_item_prob * self.prior_denominator
                )
            else:
                prior_item_item_numerator = 0.0

            n += self.prior_denominator  # to smooth n
            a += self.prior_numerator  # to smooth a

            # making the .node attribution, which will hold the prob distributions for nodes

            d = DataFrame(a / n, columns=['true'])
            d['false'] = 1 - d['true']
            d = d[['false', 'true']]

            vals = array([0, 1])
            self.node = dict()
            for kv in d.iterrows():
                self.node[kv[0]] = Pot(
                    DataFrame({kv[0]: vals, 'pval': kv[1]}, columns=[kv[0], 'pval'])
                )

            # making the .edge attribution, which will hold the likelihood distributions for pairs of nodes

            ba_count += prior_item_item_numerator
            d = DataFrame((a - ba_count) / (n - a), columns=['10'])
            d['00'] = 1 - d['10']
            d['11'] = ba_count / a
            d['01'] = 1 - d['11']
            d = d[['00', '01', '10', '11']]

            b_vals = array([0, 0, 1, 1])
            a_vals = array([0, 1, 0, 1])
            self.edge = defaultdict(dict)
            for kv in d.iterrows():
                self.edge[kv[0][0]].update(
                    {
                        kv[0][1]: Pot(
                            DataFrame(
                                {kv[0][0]: b_vals, kv[0][1]: a_vals, 'pval': kv[1]},
                                columns=[kv[0][0], kv[0][1], 'pval'],
                            )
                        )
                    }
                )
        # a dict that will default to the self.pb if it can't find a requested key
        self.post_pot = None
        self.initialize_post_pot()

    def initialize_post_pot(self):
        self.post_pot = DictDefaultDict(self.node)

    def post_prob_sorted_vars(self):
        t = [
            {'var': k, 'prob': self.post_pot[k].pval_of({k: 1})}
            for k in list(self.node.keys())
        ]
        return (
            pd.DataFrame(t, columns=['var', 'prob'])
            .sort('prob', ascending=False)
            .reset_index(drop=True)
        )
