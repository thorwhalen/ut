__author__ = 'thor'

import numpy as np


def trailing_underscore_attributes(obj):
    return [k for k in obj.__dict__.keys() if k[-1] == '_']


def get_model_attributes(model):
    if isinstance(model, (list, np.ndarray, tuple, dict, float, int)):
        return model
    else:
        return {model.__class__.__name__:
                {k: get_model_attributes(model.__getattribute__(k)) for k in trailing_underscore_attributes(model)}}

