"""Shapley value analysis"""
__author__ = 'thor'

from numpy import unique, concatenate, sort
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict, defaultdict
import string
import random as rnd
import itertools

from scipy.misc import factorial

from ut.util.uiter import all_subsets_of, powerset

from ut.stats.bin_est.set_est import Shapley as Shapley_1

# from ut.daf.manip import rollin_col


def _coalition_of(iter_of_items):
    return tuple(unique(iter_of_items))


def compute_shapley_values_from_coalition_values(
    coalition, normalize=False, verbose=False
):
    return compute_shapley_values_from_coalition_values_using_formula(
        coalition, normalize=normalize, verbose=verbose
    )


def compute_shapley_values_from_coalition_values_using_formula(
    coalition_values, normalize=False, verbose=False
):
    """
    Computes the Shapley values of a game specified by coalition values.
    See https://en.wikipedia.org/wiki/Shapley_value.
    :param coalition_values: The definition of the game (a dict of values of coalitions of players)
    :param normalize: True or [False]: Whether to normalize the Shapley values so they sum to 1
    :param verbose: True or [False]: Whether to print info while computing the Shapley values
    :return: Shapley values of the game specified by coalition_values
    """
    players = _universe_set_of_keys_of_dict(coalition_values)
    n = len(players)
    factorial_n = float(factorial(n))
    if verbose:
        print(('Normalizing factor: {}'.format(factorial_n)))

    def _shapley_unnormalized_weight(s):
        return factorial(s) * factorial(
            n - s - 1
        )  # all possible permutations of players before and after

    coalition_values = defaultdict(float, coalition_values)
    # print coalition_values

    shapley_values = dict()
    for player in players:
        if verbose:
            print(('\n-------------------- {} ----------------------'.format(player)))
        shapley_values[player] = 0.0
        for s in map(_coalition_of, powerset(players - {player})):
            shapley_values[player] += _shapley_unnormalized_weight(len(s)) * (
                coalition_values[_coalition_of(list(set(s).union({player})))]
                - coalition_values[s]
            )
            if verbose:
                weight = _shapley_unnormalized_weight(len(s))
                s_with_player = coalition_values[
                    _coalition_of(list(set(s).union({player})))
                ]
                s_alone = coalition_values[s]
                print(
                    (
                        '... contributed {} * ({} - {}) = {} \tto {} \t(running sum is {})'.format(
                            weight,
                            s_with_player,
                            s_alone,
                            weight * (s_with_player - s_alone),
                            _coalition_of(list(set(s).union({player}))),
                            shapley_values[player],
                        )
                    )
                )
        shapley_values[
            player
        ] /= factorial_n  # normalize according to all possible permutations

    if normalize:
        return _normalize_dict_values(shapley_values)
    else:
        return shapley_values


def _shapley_weight(s, n):
    return (factorial(s) * factorial(n - s - 1)) / float(factorial(n))


def compute_shapley_values_from_coalition_values_01(
    coalition_values, normalize=False, verbose=False
):
    _complete_missing_coalitions_with_zero_valued_coalitions_in_place(coalition_values)
    coalition_values = pd.DataFrame(
        index=list(coalition_values.keys()),
        data=list(coalition_values.values()),
        columns=['value'],
    )
    se = Shapley_1(coalition_values, success='value')
    se.change_type_of_d_index(tuple)
    shapley_values = se.compute_shapley_values()
    if normalize:
        return _normalize_dict_values(shapley_values)
    else:
        return shapley_values


def compute_shapley_values_from_unit_valued_sequences(sequences, normalize=False):
    dm = ShapleyDataModel()
    dm.absorb_coalition_obs(sequences)
    coalition_values = dm.coalition_values()

    return compute_shapley_values_from_coalition_values(
        coalition_values, normalize=normalize
    )


# def compute_shapley_values_from_valued_sequences(sequence_and_value_dict, normalize=False):
#     dm = ShapleyDataModel()
#     dm.absorb_sequence_into_coalition_obs(sequence_and_value_dict)
#     coalition_values = dm.coalition_values()
#
#     return compute_shapley_values_from_coalition_values(coalition_values, normalize=normalize)


def _normalize_dict_values(d):
    value_sum = float(np.sum(list(d.values())))
    return {k: v / value_sum for k, v in list(d.items())}


def all_proper_subsets_iterator(superset):
    return itertools.chain(
        *map(
            lambda subset_size: itertools.combinations(superset, subset_size),
            list(range(1, len(superset))),
        )
    )


def all_subsets_or_eq_iterator(superset):
    return itertools.chain(
        *map(
            lambda subset_size: itertools.combinations(superset, subset_size),
            list(range(1, len(superset) + 1)),
        )
    )


def all_superset_iterator(subset, universe_set):
    subset = set(subset)
    remaining_set = set(universe_set).difference(subset)
    return map(
        lambda x: tuple(subset.union(x)), all_subsets_or_eq_iterator(remaining_set)
    )


def _universe_set_of_keys_of_dict(d):
    return set(itertools.chain(*list(d.keys())))


def _complete_missing_coalitions_with_zero_valued_coalitions_in_place(
    coalition_values, universe_set=None
):
    """
    complete coalition_contributions with missing combinations (assigning 0.0 to them)
    """
    if universe_set is None:
        universe_set = set(itertools.chain(*list(coalition_values.keys())))
    superset = ShapleyDataModel.coalition_of(list(universe_set))
    coalition_values[superset] = coalition_values.get(superset, 0.0)
    for subset in map(_coalition_of, all_proper_subsets_iterator(universe_set)):
        coalition_values[subset] = coalition_values.get(subset, 0.0)


class ShapleyDataModel(object):
    def __init__(self, data=None, data_type=None):
        """
        Inputs:
            * data: data used to make the coalition values.
            * data_type: type of data, either:
                - 'coalition_obs': data is the counter (a dict) of coalition_obs directly
                - 'coalition_obs_collection': a coalition_obs dict to be added to the existing
                - 'item_collections': an iterator of sequences to absorbe to make the coalition_obs
        """
        self.coalition_obs = Counter()
        self.item_list = []
        self._coalition_size_map = None
        if data is not None:
            # if data_type not given, determine
            if data_type is None:
                if isinstance(data, Counter):
                    data_type = 'coalition_obs'
                else:
                    data_type = 'item_collections'

            # according to type, process and set data
            if data_type == 'coalition_obs':
                self.coalition_obs = data
            elif data_type == 'coalition_obs_collection':
                self.absorb_coalition_obs(data)
            elif data_type == 'item_collections':
                for d in data:
                    self.absorb_sequence_into_coalition_obs(d)

    @staticmethod
    def coalition_of(iter_of_items):
        return tuple(unique(iter_of_items))

    def absorb_sequence_into_coalition_obs(self, seq):
        """
        Updates the self.coalition_obs with the input coalition (a list of items)
        """
        self.coalition_obs.update([self.coalition_of(seq)])
        return self

    def absorb_coalition(self, collection_of_items_of_single_coalition):
        """
        Updates the self.coalition_obs with the input coalition (a list of items)
        """
        raise DeprecationWarning(
            'absorb_coalition is being deprecated. Use absorb_sequence_into_coalition_obs() instead'
        )
        self.coalition_obs.update(
            [self.coalition_of(collection_of_items_of_single_coalition)]
        )
        return self

    def absorb_coalition_obs(self, coalition_obs_data):
        """
        Updates the self.coalition_obs with the input dict of {coalition: obs_value}
        """
        try:
            for coalition, value in coalition_obs_data.items():
                self.absorb_coalition_and_value(coalition, value)
        except AttributeError:
            try:
                for seq in coalition_obs_data:
                    self.absorb_sequence_into_coalition_obs(seq)
            except TypeError:
                for seq in coalition_obs_data:
                    self.absorb_coalition_and_value(seq['coalition'], seq['value'])
        return self
        # coalition_obs_dict = \
        #     {self.coalition_of(coalition): value for coalition, value in coalition_obs_dict.iteritems()}
        # self.coalition_obs.update(coalition_obs_dict)
        # self.absorb_coalition_and_value(coalition_obs_dict.keys()[0], coalition_obs_dict.values()[0])

    def absorb_coalition_and_value(self, coalition, value):
        """
        Updates the self.coalition_obs with the input dict of coalition: obs_value
        """
        self.coalition_obs.update({self.coalition_of(coalition): value})
        return self

    def coalition_values(self, coalition_obs=None, verbose=False):
        """
        Computes the coalition_values from coalition_obs (counts or other values).
        To do this, we accumulate the counts of all subsets of each unique coalition.
        """
        if coalition_obs is None:
            coalition_obs = self.coalition_obs

        coalition_contributions = Counter(coalition_obs)

        if verbose:
            print(coalition_contributions)

        universe_set = set(self.mk_item_list(coalition_obs=coalition_obs))

        for coalition, count in coalition_obs.items():  # for every coalition
            # ... get all non-empty strict subsets of this coalition, and assign the mother coalition count
            superset_counts = {
                self.coalition_of(sub_coalition): count
                for sub_coalition in all_superset_iterator(coalition, universe_set)
            }
            # ... update the coalition_values counter with these counts
            coalition_contributions.update(superset_counts)

            if verbose:
                print(
                    (
                        '  after {} contributions:\n     {}'.format(
                            coalition, coalition_contributions
                        )
                    )
                )

        # # complete coalition_contributions with missing combinations (assigning 0.0 to them)
        # _complete_missing_coalitions_with_zero_valued_coalitions_in_place(coalition_contributions)

        return coalition_contributions

    def coalition_size_map(self):
        if not self._coalition_size_map:
            self._coalition_size_map = defaultdict(dict)
            for coalition, count in self.coalition_obs.items():
                self._coalition_size_map[len(coalition)].update({coalition: count})
        self._coalition_size_map = OrderedDict(
            sorted(list(self._coalition_size_map.items()), key=lambda t: t[0])
        )
        return self._coalition_size_map

    def mk_poset(self):
        d = defaultdict(list)
        _coalition_size_map = self.coalition_size_map()
        coalition_sizes = sorted(_coalition_size_map.keys())
        # TODO: Finish, if necessary

    def mk_item_list(self, coalition_obs=None):
        if coalition_obs is None:
            coalition_obs = self.coalition_obs
        item_list = unique(concatenate(list(coalition_obs.keys())))
        self.item_list = item_list
        return item_list


def _test_shapley_data_model():
    list_of_coalitions = [
        ['A', 'B', 'C'],
        ['A', 'C', 'B'],
        ['B', 'A', 'C'],
        ['A', 'A', 'B', 'C'],
        ['C', 'A'],
        ['B', 'C'],
        ['C', 'B'],
        ['C', 'B'],
        ['A'],
    ]
    dm = ShapleyDataModel()  # initialize the data model

    for coalition in list_of_coalitions:  # count the coalitions
        dm.absorb_sequence_into_coalition_obs(coalition)
    assert dm.coalition_obs == Counter(
        {('A', 'B', 'C'): 4, ('B', 'C'): 3, ('A',): 1, ('A', 'C'): 1}
    ), 'Unexpected result for dm.coalition_obs'

    print('All good in _test_shapley_data_model')


def rand_shapley_values(items=3):
    if isinstance(items, int):
        items = ','.join(string.ascii_uppercase[:items]).split(',')
    if isinstance(items, list):
        items = {items[i]: 2 ** i for i in range(len(items))}
    return items


class LinearValuedCoalitionGenerator(object):
    def __init__(self, shapley_values=3, normalize=False):
        shapley_values = shapley_values or 3
        if not isinstance(shapley_values, dict):
            shapley_values = rand_shapley_values(items=shapley_values)
        self.shapley_values = shapley_values
        if normalize:
            self.shapley_values = _normalize_dict_values(self.shapley_values)

    @staticmethod
    def coalition_of(coalition):
        return tuple(sort(coalition))

    def coalition_value(self, coalition):
        return sum([self.shapley_values[item] for item in coalition])

    def rand_coalition(self):
        return self.coalition_of(
            rnd.sample(
                list(self.shapley_values.keys()),
                rnd.randint(1, len(self.shapley_values)),
            )
        )

    def rand_coalition_obs(self):
        coalition = self.rand_coalition()
        return {coalition: self.coalition_value(coalition)}

    def rand_coalition_obs_cum(self, n_draws=None):
        n_draws = n_draws or len(self.shapley_values) / 2
        coalition_obs = Counter()
        for x in itertools.starmap(
            self.rand_coalition_obs, itertools.repeat([], n_draws)
        ):
            coalition_obs.update(x)
        return coalition_obs

    def coalition_values(self):
        return {
            self.coalition_of(coalition): self.coalition_value(coalition)
            for coalition in all_subsets_of(
                list(self.shapley_values.keys()), include_empty_set=False
            )
        }


# class ShapleyDataModel_old(object):
#     def __init__(self, item_seperator=','):
#         """
#         Inputs:
#             * item_seperator will be used to construct string hashes from lists.
#             You should choose a character that never shows up in the items, or you'll get problems.
#         Other attributes:
#             * coalition_obs is a Counter of coalitions
#             * coalition_values is also a Counter of coalitions, but it counts not only
#             the coalition_obs, but all non-empty subsets of the latter.
#         """
#         self.coalition_obs = Counter()
#         self.coalition_values = None
#         self.item_seperator = item_seperator
#         self.contribution_df = None
#         self.item_list = []
#
#     def absorb_coalition(self, coalition):
#         """
#         Updates the self.coalition_obs with the input coalition (a list of items)
#         """
#         self.coalition_obs.update([self._list_to_key(coalition)])
#
#     def mk_coalition_size_map(self):
#
#         d = defaultdict(list)
#         for coalition, count in self.coalition_obs.iteritems():
#             d[len(self._key_to_list(coalition))].append({coalition: count})
#         return d
#
#     def mk_coalition_contributions(self, verbose=False):
#         """
#         Computes the self.coalition_values attribute.
#         To do this, we accumulate the counts of all subsets of each unique coalition.
#         """
#         # init with coalition_obs
#         self.coalition_values = Counter(self.coalition_obs)
#         if verbose:
#             print(self.coalition_values)
#         for coalition, count in self.coalition_obs.iteritems():  # for every coalition
#             # get list corresponding to the key
#             coalition = self._key_to_list(coalition)
#             # get all non-empty strict subsets of this list,
#             # and assign the mother coalition count
#             subset_counts = \
#                 {self._list_to_key(sub_coalition): count
#                  for sub_coalition in all_proper_subsets_iterator(coalition)}
#             # update the coalition_values counter with these counts
#             self.coalition_values.update(subset_counts)
#             if verbose:
#                 print("  after {} contributions:\n     {}" \
#                       .format(coalition, self.coalition_values))
#
#     def mk_item_list(self):
#         self.item_list = list(unique(self.item_seperator.join(dm.coalition_obs.keys()) \
#                                      .split(self.item_seperator)))
#
#         # def all_supersets_iterator(self, subset):
#
#     #         subset = dm
#
#     def mk_contribution_df(self):
#         self._fill_counters()
#         self.contribution_df = \
#             pd.DataFrame(index=self.coalition_values.keys(), columns=dm.item_list)
#         for coalition in self.contribution_df.index.values:
#             print self._remove_and_remain_dicts(coalition)
#             for rr in self._remove_and_remain_dicts(coalition):
#                 # the contribution of each item is the total contribution
#                 # minus what the contribution would be without this item
#                 contribution = \
#                     self.coalition_values[coalition] \
#                     - self.coalition_values[rr['remaining']]
#                 # enter this in the contribution_df
#                 self.contribution_df.loc[coalition, rr['removed']] = contribution
#
#     def _fill_counters(self):
#         """
#         adds missing item combinations to counters, giving them 0 count
#         """
#         self.mk_item_list()
#         zero_counts = {k: 0 for k in itertools.imap(self._list_to_key,
#                                                     all_proper_subsets_iterator(self.item_list))
#                        }
#         self.coalition_obs.update(zero_counts)
#         self.coalition_values.update(zero_counts)
#
#     def _list_to_key(self, coalition):
#         """
#         Transforms a list of strings to a comma (or item_seperator) separated string
#         of unique items of the input list.
#         """
#         return self.item_seperator.join(unique(coalition))
#
#     def _key_to_list(self, coalition_key):
#         """
#         Inverse of _list_to_key:
#         Returns a list from a character (item_seperator) seperated string of items.
#         """
#         return coalition_key.split(self.item_seperator)
#
#     def _remove_and_remain_dicts(self, superset):
#         """
#         Returns a list of {removed, remaining} dicts listing all (keys of) superset - item
#         sets for every item in superset.
#         Returns an empty list if the input superset has only one element.
#         Example:
#             self._remove_and_remain_dicts('A,B,C')
#         returns
#             [{'remaining': 'B,C', 'removed': 'A'},
#              {'remaining': 'A,B', 'removed': 'C'},
#              {'remaining': 'A,C', 'removed': 'B'}]
#         """
#         superset = set(self._key_to_list(superset))
#         if len(superset) > 1:
#             return [{'removed': x,
#                      'remaining': self._list_to_key(
#                          list(superset.difference(x)))}
#                     for x in superset]
#         else:
#             return list()  # return empty list if superset has only one element
#
#
# def _test_shapley_data_model():
#     list_of_coalitions = [['A', 'B', 'C'], ['A', 'C', 'B'], ['B', 'A', 'C'], ['A', 'A', 'B', 'C'],
#                           ['C', 'A'], ['B', 'C'], ['C', 'B'], ['C', 'B'], ['A']]
#     dm = ShapleyDataModel_old()  # initialize the data model
#
#     for coalition in list_of_coalitions:  # count the coalitions
#         dm.absorb_coalition(coalition)
#     assert dm.coalition_obs == Counter({'A,B,C': 4, 'B,C': 3, 'A': 1, 'A,C': 1}), \
#         "Unexpected result for dm.coalition_obs"
#
#     dm.mk_coalition_contributions()
#     assert dm.coalition_values \
#            == Counter({'C': 8, 'B': 7, 'B,C': 7, 'A': 6, 'A,C': 5, 'A,B,C': 4, 'A,B': 4}), \
#         "Unexpected result for dm.coalition_values"
#
#     print("All good in _test_shapley_data_model")
