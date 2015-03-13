__author__ = 'thor'

import numpy as np
from decimal import Decimal

def rand_numbers_summing_to_one(n_numbers, granularity=0.01):
    n_choices = 1.0 / granularity
    assert round(n_choices) == int(n_choices), "granularity must be an integer divisor of 1.0"
    x = np.linspace(granularity, 1.0 - granularity, n_choices - 1)
    x = sorted(x[np.random.choice(range(1, len(x)), size=n_numbers - 1, replace=False)])
    x = np.concatenate([[0.0], x, [1.0]])
    x = np.diff(x)
    x = np.array([Decimal(xi).quantize(Decimal(str(granularity))) for xi in x])
    return x



