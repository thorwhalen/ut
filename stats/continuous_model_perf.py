__author__ = 'thor'


from numpy import *


def mk_mean_model(y):
    """
    The mean model (often used as a benchmark)
    Returns a function returning the mean of the values of y as the model's estimate (no matter what the input x is)
    """
    mean_y = mean(y)
    return lambda x: mean_y



