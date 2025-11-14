from sklearn.base import BaseEstimator
from collections import Counter
import pandas as pd
from numpy import sum, nan, isnan
from ut.util.uiter import window


class NextElementPredictor(BaseEstimator):
    def predict(self, seqs):
        preds = self.predict_proba(seqs)
        return [max(pred, key=lambda key: pred[key]) for pred in preds]

    def predict_proba(self, seqs):
        return list(map(self._predict_proba_conditioned_on_recent_subseq, seqs))

    def _predict_proba_conditioned_on_recent_subseq(self, recent_subseq):
        raise NotImplementedError('Need to implement this method')


class MarkovNextElementPred(NextElementPredictor):

    _list_of_attributes_to_display = [
        'markov_window',
        'empty_element',
        'keep_stats_in_memory',
    ]

    def __init__(self, markov_window=2, empty_element=-1, keep_stats_in_memory=True):
        self.markov_window = markov_window
        self.keep_stats_in_memory = keep_stats_in_memory
        self.empty_element = empty_element
        self._empty_element_padding = [empty_element] * (self.markov_window - 1)

    @property
    def total_tuple_count(self):
        """
        :return: Number of observed window tuples (sum of values in self.snip_tuples_counter_)
        """
        if self.total_tuple_count_ is not None:
            return self.total_tuple_count_
        else:
            total_tuple_count_ = sum(self.snip_tuples_counter_.values())
            if self.keep_stats_in_memory:
                self.total_tuple_count_ = total_tuple_count_
            return total_tuple_count_

    @property
    def pair_prob(self):
        """
        :return: Series of probabilities (unsmoothed count ratios) indexed by snip pairs
        """
        if self.pair_prob_ is not None:
            return self.pair_prob_
        else:
            pair_prob_ = pd.Series(self.snip_tuples_counter_) / self.total_tuple_count
            if self.keep_stats_in_memory:
                self.pair_probs_ = pair_prob_
            return pair_prob_

    @property
    def element_prob(self):
        """
        :return: Series of snips probabilities (unsmoothed count ratios)
        """
        if self.element_prob_ is not None:
            return self.element_prob_
        else:
            element_prob_ = self.pair_prob * self.total_tuple_count
            element_prob_ = element_prob_.groupby(level=0).sum()
            element_prob_ = element_prob_.drop(labels=self.empty_element)
            # element_prob_ = element_prob_.iloc[
            #     element_prob_.index.get_level_values(level=0) != self.empty_element]
            element_prob_ /= element_prob_.sum()
            if self.keep_stats_in_memory:
                self.element_prob_ = element_prob_
            return element_prob_

    @property
    def conditional_prob(self):
        """
        :return: Series of probabilities of last element (level) conditional on previous ones (including empty elements)
        """
        if self.conditional_prob_ is not None:
            return self.conditional_prob_
        else:
            conditional_prob_ = self._drop_empty_elements_of_sr(
                self.pair_prob, levels=[self.markov_window - 1]
            )
            conditional_levels = list(range(self.markov_window - 1))
            conditional_prob_ = conditional_prob_.div(
                conditional_prob_.groupby(level=conditional_levels).sum(), level=0
            )  # TODO: Only works for two levels
            if self.keep_stats_in_memory:
                self.conditional_prob_ = conditional_prob_
            return conditional_prob_

    @property
    def initial_element_prob(self):
        """
        :return: Series of snips probabilities (unsmoothed count ratios)
        """
        if self.initial_element_prob_ is not None:
            return self.initial_element_prob_
        else:
            initial_element_prob_ = self.pair_prob.xs(
                self.empty_element, level=0, drop_level=True
            )
            initial_element_prob_ /= initial_element_prob_.sum()
            if self.keep_stats_in_memory:
                self.initial_element_prob_ = initial_element_prob_
            return initial_element_prob_

    def fit(self, snips_list):
        # reset anything previously learned
        self._initialize_params()
        return self.partial_fit(snips_list)

    def partial_fit(self, snips_list):
        if not {'snip_tuples_counter_'}.issubset(list(self.__dict__.keys())):
            self._initialize_params()
        for snips in snips_list:
            self._partial_fit_of_a_single_snips(snips)
        return self

    def _initialize_params(self):
        """
        Initializes model params (the snip_tuples_counter_, etc.)
        :return: None
        """
        self.snip_tuples_counter_ = Counter()
        self._reset_properties()

    def _reset_properties(self):
        """
        Resets some properties that depend on snip_tuples_counter_ to be computed (is used when the later changes)
        These will be recomputed when requested.
        :return: None
        """
        self.total_tuple_count_ = None
        self.pair_prob_ = None
        self.element_prob_ = None
        self.initial_element_prob_ = None
        self.conditional_prob_ = None

    def _partial_fit_of_a_single_snips(self, snips):
        self._reset_properties()
        self.snip_tuples_counter_.update(
            window(
                self._empty_element_padding + list(snips) + self._empty_element_padding,
                n=self.markov_window,
            )
        )

    def _drop_empty_elements_of_sr(self, sr, levels=None, renormalize=False):
        if levels is None:
            levels = list(range(self.markov_window))
        for level in levels:
            sr = sr.drop(labels=self.empty_element, level=level)
        if renormalize:
            sr /= sr.sum()
        return sr

    def _predict_proba_conditioned_on_recent_subseq(self, recent_subseq):
        pass

    def __repr__(self):
        d = {
            attr: getattr(self, attr)
            for attr in self._list_of_attributes_to_display
            if attr in self.__dict__
        }
        d['total_tuple_count'] = self.total_tuple_count
        return self.__class__.__name__ + '\n' + str(d)
