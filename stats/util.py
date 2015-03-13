__author__ = 'thor'

from numpy import *


def _df_picker(df, x_cols, y_col):
    return df[x_cols].as_matrix(), df[[y_col]].as_matrix()


def df_picker_data_prep(x_cols, y_col):
    return lambda df: _df_picker(df, x_cols, y_col)


def binomial_probs_to_multinomial_probs(binomial_probs):
    multinomial_probs = zeros((len(binomial_probs), 2))
    multinomial_probs[:, 1] = binomial_probs
    multinomial_probs[:, 0] = 1 - multinomial_probs[:, 1]
    return multinomial_probs


def multinomial_probs_to_binomial_probs(multinomial_probs):
    return multinomial_probs[:, 1]


def normalize_to_one(arr):
    arr = array(arr)
    return arr / float(sum(arr))




