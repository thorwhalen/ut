__author__ = 'thor'

import pandas as pd
import numpy as np

from collections import Counter, defaultdict
from itertools import islice, chain
import matplotlib.pylab as plt


class Markov(object):
    def __init__(self, cond_probs, initial_probs, states=None, t_name=None, t_plus_1_name=None):
        self.initial_probs = initial_probs
        self.cond_probs = cond_probs
        if states is None:  # take the states of initial probs, sorted by descending order of probability
            states = list(self.initial_probs.index.values)
            more_states = set(self.cond_probs.index.values).union(self.cond_probs.columns.values)
            for extra_state in set(more_states).difference(states):
                states.append(extra_state)
                self.initial_probs.loc[extra_state] = 0.0
            states = self.initial_probs.sort(inplace=False, ascending=False).index.values

        self.labels = states
        self.initial_probs = self.initial_probs[states]

        self.cond_probs = self.cond_probs.loc[states, states].fillna(0.0)

        if t_name is None:  # given the name of the columns, or 't' if columns have no name
            t_name = self.cond_probs.columns.name or 't'
        self.t_name = t_name
        self.cond_probs.columns.name = t_name

        if t_plus_1_name is None:  # given the name of the index, or 't+1' if index have no name
            t_plus_1_name = self.cond_probs.index.name or 't+1'
        self.t_plus_1_name = t_plus_1_name
        self.cond_probs.index.name = t_plus_1_name

    def plot_matrix_prod(self, n=1, display_transpose=False):
        cond_prob_matrix = np.matrix(self.cond_probs.as_matrix())
        cond_prob_matrix = cond_prob_matrix ** n
        if display_transpose:
            cond_prob_matrix = cond_prob_matrix.T
        plt.matshow(cond_prob_matrix);
        ax = plt.gca()
        plt.xticks(list(range(len(self.labels))))
        ax.set_xticklabels(self.labels, rotation=90)
        plt.yticks(list(range(len(self.labels))))
        ax.set_yticklabels(self.labels)
        plt.grid('off');

    @staticmethod
    def from_sequences(seqs, **kwargs):
        initial_probs = Markov.seqs_to_initial_probs(seqs)

        cond_probs = Markov.seqs_to_pair_count_df(seqs)
        cond_probs = cond_probs.divide(cond_probs.sum(axis=0), axis='columns')

        return Markov(cond_probs=cond_probs, initial_probs=initial_probs, **kwargs)

    @staticmethod
    def seqs_to_initial_probs(seqs):
        initial_probs = Counter([seq[0] for seq in seqs])
        initial_probs = pd.Series(initial_probs)
        initial_probs = initial_probs / initial_probs.sum()
        return initial_probs

    @staticmethod
    def seqs_to_pair_count_df(seqs):
        event_pair_counts = Counter(chain(*map(_sliding_window_iter, seqs)))
        pair_count_df = pd.DataFrame([{'t': k[0], 't+1': k[1], 'count': v}
                                  for k, v in event_pair_counts.items()])
        pair_count_df = pair_count_df.set_index(['t', 't+1']).sort()
        pair_count_df = pair_count_df['count'].unstack('t')
        return pair_count_df

    @staticmethod
    def from_markov_counts(mc, states=None, t_name='t', t_plus_1_name='t+1', prior_pair_count=0.0):
        initial_probs = pd.Series(mc.initial_counts).sort(inplace=False, ascending=False)
        initial_probs /= np.sum(initial_probs)

        cond_probs = [{t_name: k[0], t_plus_1_name: k[1], 'count': v} for k, v in mc.pair_counts.items()]
        cond_probs = pd.DataFrame(cond_probs)\
            .set_index([t_name, t_plus_1_name])['count']\
            .unstack(t_name)\
            .fillna(0.0)
        cond_probs += prior_pair_count
        cond_probs = cond_probs.divide(cond_probs.sum(axis=0), axis='columns')
        return Markov(cond_probs=cond_probs, initial_probs=initial_probs)


class MarkovCounts(object):
    def __init__(self):
        self.initial_counts = Counter()
        self.pair_counts = Counter()

    def add_sequence(self, seq):
        self.initial_counts.update([seq[0]])
        self.pair_counts.update(_sliding_window_iter(seq))


class IndexedMarkovCounts(object):
    def __init__(self):
        self.markov_counts = defaultdict(lambda: MarkovCounts())

    def add_sequence(self, index, seq):
        self.markov_counts[index].add_sequence(seq)

    def __getstate__(self):
        return {
            'markov_counts': dict(self.markov_counts)
        }

    def __setstate__(self, state):
        self.markov_counts = defaultdict(lambda: MarkovCounts(), state['markov_counts'])

    # def __getstate__(self):
    #     return {
    #         'initial_counts': dict(self.initial_counts),
    #         'pair_counts': dict(self.pair_counts),
    #     }
    #
    # def __setstate__(self, state):
    #     self.initial_counts = defaultdict(lambda: Counter(), state['initial_counts'])
    #     self.pair_counts = defaultdict(lambda: Counter(), state['pair_counts'])


class MultipleMarkovCounts(object):
    def __init__(self):
        self.initial_counts = defaultdict(lambda: Counter())
        self.pair_counts = defaultdict(lambda: Counter())

    def add_sequence(self, index, seq):
        self.initial_counts[index].update([seq[0]])
        self.pair_counts[index].update(_sliding_window_iter(seq))

    def __getstate__(self):
        return {
            'initial_counts': dict(self.initial_counts),
            'pair_counts': dict(self.pair_counts),
        }

    def __setstate__(self, state):
        self.initial_counts = defaultdict(lambda: Counter(), state['initial_counts'])
        self.pair_counts = defaultdict(lambda: Counter(), state['pair_counts'])



def _sliding_window_iter(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result