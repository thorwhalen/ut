__author__ = 'thor'

import pandas as pd
import itertools


def equality_counts(df):
    """
    returns a Series indexed by every True/False combination of observed equalities
    of the columns of df, along with the count of this combination.
    Example:

        criteria       source
        False          False      5063
                       True         89
        True           False     23936
                       True       1293

    means that out of all pairs of the rows (i,j) (i != j)of df, having columns "criteria" and
    "source",

        5063 of these pairs had both criteria_i != criteria_j and source_i != source_j,
        89 of these pairs had both criteria_i != criteria_j and source_i == source_j,
        23936 of these pairs had both criteria_i == criteria_j and source_i != source_j,
        1293 of these pairs had both criteria_i == criteria_j and source_i == source_j

    """
    eq_counts = pd.DataFrame()
    for i, j in itertools.combinations(xrange(len(df)), 2):
        eq_counts = pd.concat([eq_counts, df.iloc[i] == df.iloc[j]], axis=1)
    eq_counts = eq_counts.T.reset_index(drop=True)
    return eq_counts.groupby(list(eq_counts.columns)).size()


