"""Machine learning utils"""
__author__ = 'thor'

import numpy as np
from json import JSONEncoder, dump, dumps
from datetime import datetime
import pickle
from scipy.sparse import issparse
import dill
from copy import deepcopy
# from types import NoneType
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


class ExtrapolateTransformation(TransformerMixin):
    """A wrapper that endows a TransformerMixin that has no transform method with one."""
    def __init__(self, transformer, extrapolator=LinearRegression()):
        self.transformer = transformer
        self.extrapolator = extrapolator

    def fit(self, X, y=None):
        try:
            transformed_X = self.transformer.fit_transform(X, y)
        except ValueError:
            y = LabelEncoder().fit_transform(y)
            transformed_X = self.transformer.fit_transform(X, y)

        if isinstance(self.extrapolator, type):
            self.extrapolator = self.extrapolator()
        self.extrapolator.fit(X, transformed_X)

        return self

    def transform(self, X):
        return self.extrapolator.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def __getattr__(self, item):
        return self.transformer.__getattribute__(item)


# default_as_is_types = (list, np.ndarray, tuple, dict, float, int)
default_as_is_types = (list, np.ndarray, tuple, dict, float, int, set, np.int32,
                       str, np.matrixlib.defmatrix.matrix, type(None))
# default_as_is_types = (list, tuple, dict, float, int, set, np.int32,
#                        basestring, NoneType)

def trailing_underscore_attributes(obj):
    return [k for k in obj.__dict__ if k[-1] == '_']


def trailing_underscore_attributes_with_include_and_exclude(obj, include=(), exclude=()):
    return [k for k in obj.__dict__ if k in include or (k[-1] == '_' and k not in exclude)]


def get_model_attributes(model,
                         include=(),
                         exclude=(),
                         model_name_as_dict_root=True,
                         as_is_types=default_as_is_types):
    """
    Export parameters of the model (or any object) to a dict.
    :param include: list of attributes to include (the function will automatically include attributes ending with
    an underscore
    :param exclude: list of attributes to exclude (to exclude underscore suffixed attributes
    :param model_name_as_dict_root: Whether to wrap the dict into a dict whose single key is the name of the model
    :param as_is_types: Types to take as is (all others will be passed on to get_model_attributes recursively
    :param version: String to include in the "version" field of the json
    :return: Nothing if dumping to json file, or the json string if argument filepath=None
    """
    if isinstance(model, as_is_types):  # if model is in as_is_types list, just return it
        return model
    elif not hasattr(model, '__dict__') or len(model.__dict__) == 0:
        states = pickle.dumps(model)
    else:
        # getting the list of attributes to save
        if isinstance(include, str):
            if include == 'all':
                if len(exclude) > 0:
                    attribute_set = [k for k in model.__dict__ if k not in exclude]
                else:
                    attribute_set = model.__dict__
            elif include == 'all_but_double_underscore':
                if len(exclude) > 0:
                    attribute_set = [k for k in model.__dict__ if k not in exclude and not k.startswith('__')]
                else:
                    attribute_set = model.__dict__
        else:
            attribute_set = trailing_underscore_attributes_with_include_and_exclude(model, include, exclude)
        # attribute_set = set(trailing_underscore_attributes(model)).union(include).difference(exclude)

        # recursion on the values that are not in as_is_types
        states = {k: get_model_attributes(getattr(model, k),  #model.__getattribute__(k),
                                          include,
                                          exclude,
                                          model_name_as_dict_root,
                                          as_is_types) for k in attribute_set}
    # wrap in model name
    if model_name_as_dict_root:
        return {model.__class__.__name__: states}
    else:
        return states


def get_model_attributes_dict_for_json(model,
                                       include=(),
                                       exclude=(),
                                       model_name_as_dict_root=True,
                                       as_is_types=default_as_is_types):
    if isinstance(model, as_is_types):
        return model
    elif isinstance(model, np.ndarray):
        return model.tolist()
    else:
        return json_friendly_dict(get_model_attributes(model,
                                    include,
                                    exclude,
                                    model_name_as_dict_root,
                                    as_is_types
                                    ))


def export_model_params_to_json(model,
                                include=(),
                                exclude=(),
                                model_name_as_dict_root=True,
                                as_is_types=default_as_is_types,
                                filepath='',
                                version=None,
                                include_date=False,
                                indent=None):
    """
    Export parameters of the model to a json file or return a json string.
    :param filepath: Filepath to dump the json string to, or "" to just return the string, or None to return the dict
    :param version: String to include in the "version" field of the json
    :return: Nothing if dumping to json file, or the json string if argument filepath=None
    """
    model_params = get_model_attributes(model,
                                        include,
                                        exclude,
                                        model_name_as_dict_root,
                                        as_is_types
                                        )
    model_params = model_params.copy()

    if include_date:
        model_params['date'] = str(datetime.now())
        if isinstance(include_date, str) and include_date == 'as string':
            model_params['date'] = str(model_params['date'])
    if version:
        model_params['version'] = version
    if filepath is not None:
        if filepath == '':
            return dumps(model_params, indent=indent, cls=NumpyAwareJSONEncoder)
        else:
            print(("Saving the centroid_model_params to {}".format(filepath)))
            dump(model_params, open(filepath, 'w'), indent=indent, cls=NumpyAwareJSONEncoder)
    else:
        return model_params


def import_model_from_spec(spec,
                           objects={}.copy(),
                           type_conversions=(),
                           field_conversions={}.copy(),
                           force_dict_wrap=False):
    if isinstance(spec, dict):
        model_dict_imported = dict()
        for k, v in spec.items():
            # print k
            if k in objects:
                obj = objects[k]
                if isinstance(obj, type):
                    obj = objects[k]()
                else:
                    obj = deepcopy(obj)

                for kk, vv in v.items():
                    setattr(obj, kk,
                            import_model_from_spec(vv, objects, type_conversions, field_conversions, force_dict_wrap))
                model_dict_imported[k] = obj
            elif k in field_conversions:
                # print k
                model_dict_imported[k] = field_conversions[k](v)
            else:
                model_dict_imported[k] = import_model_from_spec(v, objects, type_conversions, field_conversions,
                                                                force_dict_wrap)
        if len(model_dict_imported) == 1 and not force_dict_wrap:
            return model_dict_imported[list(model_dict_imported.keys())[0]]
        else:
            return model_dict_imported
    else:
        if spec == 'kd_tree':
            print(spec)
        if type_conversions:
            for _type, converter in type_conversions:
                if isinstance(spec, _type):
                    return converter(spec)
        if isinstance(spec, str) and spec.startswith('cdill.dill'):
            return dill.loads(spec)
        else:
            return spec


def json_friendly_dict(obj):
    if isinstance(obj, dict):
        return {k: json_friendly_dict(v) for k, v in obj.items()}
    elif hasattr(obj, 'tolist') and callable(obj.tolist):
        return obj.tolist()
    elif hasattr(obj, 'to_list') and callable(obj.to_list):
        return obj.to_list()
    elif isinstance(obj, np.matrixlib.defmatrix.matrix):
        return list(np.array(obj))
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif issparse(obj):
        tt = obj.tocoo()
        ttt = tt.nonzero()
        return list(zip(ttt[0], ttt[1], tt.data))
    else:
        return obj


class NumpyAwareJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            super(self.__class__, self).default(self, obj)
        except TypeError as e:
            if isinstance(obj, np.matrixlib.defmatrix.matrix):
                return list(np.array(obj))
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif hasattr(obj, 'tolist') and callable(obj.tolist):
                return obj.tolist()
            elif hasattr(obj, 'to_list') and callable(obj.to_list):
                return obj.to_list()
            elif issparse(obj):
                tt = obj.tocoo()
                ttt = tt.nonzero()
                return list(map(list(zip(ttt[0], ttt[1], tt.data))))
            else:
                return list(obj)


# if __name__ == "__main__":
#     from ut.ml import utils as mlutils
#     import pickle
#     import json
#     from oto.models.centroid_smoothing import CentroidSmoothing
#     from sklearn.cluster import SpectralClustering, AgglomerativeClustering
#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.decomposition import PCA
#     # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#     from sklearn.lda import LDA
#     from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
#     from numpy import array, allclose, random
#
#     from ut.ml.sk.discriminant_analysis import FLDA, FldaLite
#
#     model = pickle.load(open('/D/Dropbox/dev/py/notebooks/soto/cs_flda.p', 'r'))
#     json_str = mlutils.export_model_params_to_json(model, include='all')
#     model_spec = json.loads(json_str)
#
#     print model
#
#     obj = mlutils.import_model_from_spec(model_spec,
#                                          objects={'CentroidSmoothing': CentroidSmoothing,
#                                                   'LinearDiscriminantAnalysis': LDA,
#                                                   'LDA': LDA,
#                                                   'FLDA': FLDA,
#                                                   'FldaLite': FldaLite,
#                                                   'StandardScaler': StandardScaler,
#                                                   'SpectralClustering': SpectralClustering,
#                                                   'KNeighborsClassifier': KNeighborsClassifier,
#                                                   'CountVectorizer': CountVectorizer(input='content',
#                                                                                      tokenizer=lambda x: x,
#                                                                                      lowercase=False),
#                                                   'TfidfTransformer': TfidfTransformer,
#                                                   'PCA': PCA},
#                                          field_conversions={'KDTree': pickle.loads,
#                                                             'tokenizer': lambda x: lambda xx: xx,
#                                                             'function': pickle.loads},
#                                          type_conversions=[(list, array)]
#                                          )
#
#     t = random.rand(10, 26)
#     p1 = model.predict_proba(t)
#     p2 = obj.predict_proba(t)
#     allclose(p1, p2)