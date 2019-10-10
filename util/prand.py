__author__ = 'thor'

import numpy as np
from decimal import Decimal
from numpy import random

def rand_numbers_summing_to_one(n_numbers, granularity=0.01):
    n_choices = 1.0 / granularity
    assert round(n_choices) == int(n_choices), "granularity must be an integer divisor of 1.0"
    x = np.linspace(granularity, 1.0 - granularity, n_choices - 1)
    x = sorted(x[np.random.choice(list(range(1, len(x))), size=n_numbers - 1, replace=False)])
    x = np.concatenate([[0.0], x, [1.0]])
    x = np.diff(x)
    x = np.array([Decimal(xi).quantize(Decimal(str(granularity))) for xi in x])
    return x


def weighted_choice(choices):
    r = random.uniform(0, sum(choices))
    upto = 0
    for i, w in enumerate(choices):
        if upto + w >= r:
            return i
        upto += w
    assert False, "Shouldn't get here"

