__author__ = 'thor'

import numpy as np


def trailing_underscore_attributes(obj):
    return [k for k in obj.__dict__.keys() if k[-1] == '_']


def get_model_attributes(model, model_name_as_dict_root=True, ignore_list=[]):
    if isinstance(model, (list, np.ndarray, tuple, dict, float, int)):
        return model
    else:
        states = {k: get_model_attributes(model.__getattribute__(k))
                  for k in set(trailing_underscore_attributes(model)).difference(ignore_list)}
        if model_name_as_dict_root:
            return {model.__class__.__name__: states}
        else:
            return states

