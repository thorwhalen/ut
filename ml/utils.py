__author__ = 'thor'

import numpy as np
from json import JSONEncoder, dump, dumps
from datetime import datetime

# default_as_is_types = (list, np.ndarray, tuple, dict, float, int)
default_as_is_types = (list, np.ndarray, tuple, dict, float, int, set, np.int32,
                       basestring, np.matrixlib.defmatrix.matrix)


def trailing_underscore_attributes(obj):
    return [k for k in obj.__dict__ if k[-1] == '_']


def get_model_attributes(model, model_name_as_dict_root=True, ignore_list=[], as_is_types=default_as_is_types):
    if isinstance(model, as_is_types):
        return model
    else:
        states = {k: get_model_attributes(model.__getattribute__(k))
                  for k in set(trailing_underscore_attributes(model)).difference(ignore_list)}
        if model_name_as_dict_root:
            return {model.__class__.__name__: states}
        else:
            return states


def get_model_attributes_dict_for_json(model, model_name_as_dict_root=True, ignore_list=[],
                                       as_is_types=default_as_is_types):
    if isinstance(model, as_is_types):
        return model
    elif isinstance(model, np.ndarray):
        return model.tolist()
    else:
        states = {k: get_model_attributes_dict_for_json(model.__getattribute__(k))
                  for k in set(trailing_underscore_attributes(model)).difference(ignore_list)}
        if model_name_as_dict_root:
            return {model.__class__.__name__: states}
        else:
            return states


def export_model_params_to_json(model, filepath='', version=None, include_date=False, indent=None):
    """
    Export parameters of the model to a json file or return a json string.
    :param filepath: Filepath to dump the json string to, or "" to just return the string, or None to return the dict
    :param version: String to include in the "version" field of the json
    :return: Nothing if dumping to json file, or the json string if argument filepath=None
    """
    model_params = get_model_attributes(model).copy()
    if include_date:
        model_params['date'] = str(datetime.now())
        if isinstance(include_date, basestring) and include_date == 'as string':
            model_params['date'] = str(model_params['date'])
    if version:
        model_params['version'] = version
    if filepath is not None:
        if filepath == '':
            return dumps(model_params, indent=indent, cls=NumpyAwareJSONEncoder)
        else:
            print("Saving the centroid_model_params to {}".format(filepath))
            dump(model_params, open(filepath, 'w'), indent=indent, cls=NumpyAwareJSONEncoder)
    else:
        return model_params


class NumpyAwareJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            super(self.__class__, self).default(self, obj)
        except TypeError:
            if isinstance(obj, np.matrixlib.defmatrix.matrix):
                return list(np.array(obj))
            elif isinstance(obj, np.int32):
                return int(obj)
            else:
                return list(obj)
