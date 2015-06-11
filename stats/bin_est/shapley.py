__author__ = 'thor'

from numpy import *
import pandas as pd
from collections import Counter, OrderedDict, defaultdict
import itertools

from ut.stats.bin_est.set_est import Shapley as Shapley_1

# from ut.daf.manip import rollin_col


def compute_shapley_values_from_coalition_contributions(coalition_values):
    coalition_values = pd.DataFrame(index=coalition_values.keys(),
                                    data=coalition_values.values(),
                                    columns=['value'])
    se = Shapley_1(coalition_values, success='value')
    se.change_type_of_d_index(tuple)
    return se.compute_shapley_values()


def all_subsets_iterator(superset):
    return itertools.chain(
        *itertools.imap(lambda subset_size: itertools.combinations(superset, subset_size),
                        range(1, len(superset))))


class ShapleyDataModel(object):
    def __init__(self, data=None, data_type=None):
        """
        Inputs:
            * item_seperator will be used to construct string hashes from lists.
            You should choose a character that never shows up in the items, or you'll get problems.
        Other attributes:
            * coalition_obs is a Counter of coalitions
            * coalition_values is also a Counter of coalitions, but it counts not only
            the coalition_obs, but all non-empty subsets of the latter.
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
                for d in data:
                    self.absorb_coalition_obs(d)
            elif data_type == 'item_collections':
                for d in data:
                    self.absorb_coalition(d)

    @staticmethod
    def coalition_of(iter_of_items):
        return tuple(unique(iter_of_items))

    def absorb_coalition(self, collection_of_items_of_single_coalition):
        """
        Updates the self.coalition_obs with the input coalition (a list of items)
        """
        self.coalition_obs.update([self.coalition_of(collection_of_items_of_single_coalition)])

    def absorb_coalition_obs(self, coalition_obs_dict):
        """
        Updates the self.coalition_obs with the input dict of coalition: obs_value
        """
        self.coalition_obs.update(coalition_obs_dict)

    def coalition_values(self, verbose=False):
        """
        Computes the self.coalition_values attribute.
        To do this, we accumulate the counts of all subsets of each unique coalition.
        """
        coalition_contributions = Counter(self.coalition_obs)

        if verbose:
            print(self.coalition_values)

        for coalition, count in self.coalition_obs.iteritems():  # for every coalition
            # ... get list corresponding to the key
            coalition = self._key_to_list(coalition)
            # ... get all non-empty strict subsets of this list, and assign the mother coalition count
            subset_counts = \
                {self._list_to_key(sub_coalition): count
                 for sub_coalition in all_subsets_iterator(coalition)}
            # ... update the coalition_values counter with these counts
            coalition_contributions.update(subset_counts)

            if verbose:
                print("  after {} contributions:\n     {}" \
                      .format(coalition, self.coalition_values))

        return coalition_contributions

    def coalition_size_map(self):
        if not self._coalition_size_map:
            self._coalition_size_map = defaultdict(dict)
            for coalition, count in self.coalition_obs.iteritems():
                self._coalition_size_map[len(coalition)].update({coalition: count})
        self._coalition_size_map = OrderedDict(sorted(self._coalition_size_map.items(), key=lambda t: t[0]))
        return self._coalition_size_map

    def mk_poset(self):
        d = defaultdict(list)
        _coalition_size_map = self.coalition_size_map()
        coalition_sizes = sorted(_coalition_size_map.keys())
        # TODO: Finish, if necessary

    def mk_item_list(self):
        self.item_list = unique(concatenate(self.coalition_obs.keys()))



def _test_shapley_data_model():
    list_of_coalitions = [['A', 'B', 'C'], ['A', 'C', 'B'], ['B', 'A', 'C'], ['A', 'A', 'B', 'C'],
                          ['C', 'A'], ['B', 'C'], ['C', 'B'], ['C', 'B'], ['A']]
    dm = ShapleyDataModel()  # initialize the data model

    for coalition in list_of_coalitions:  # count the coalitions
        dm.absorb_coalition(coalition)
    assert dm.coalition_obs == Counter({('A', 'B', 'C'): 4, ('B', 'C'): 3, ('A',): 1, ('A', 'C'): 1}), \
        "Unexpected result for dm.coalition_obs"


    print("All good in _test_shapley_data_model")

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
#                  for sub_coalition in all_subsets_iterator(coalition)}
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
#                                                     all_subsets_iterator(self.item_list))
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
