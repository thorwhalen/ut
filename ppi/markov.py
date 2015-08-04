__author__ = 'thor'

import pandas as pd
from collections import Counter
from itertools import islice, chain, imap


class Markov(object):
    def __init__(self, cond_probs, initial_probs, states=None, t_name=None, t_plus_1_name=None):
        self.initial_probs = initial_probs
        self.cond_probs = cond_probs
        if states is None:  # take the states of initial probs, sorted by descending order of probability
            states = list(self.index.values)
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
        event_pair_counts = Counter(chain(*imap(_sliding_window_iter, seqs)))
        pair_count_df = pd.DataFrame([{'t': k[0], 't+1': k[1], 'count': v}
                                  for k, v in event_pair_counts.iteritems()])
        pair_count_df = pair_count_df.set_index(['t', 't+1']).sort()
        pair_count_df = pair_count_df['count'].unstack('t')
        return pair_count_df



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