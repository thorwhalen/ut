__author__ = 'thor'

import pystan
import pickle
from hashlib import md5
import re
import os

stan_models_directory_paths = [
    './',
    './data/',
    '../data/'
]

# this_files_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

def stan_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    # retrieve or make model
    sm = get_cached_model_or_make_and_cache_it(model_code, model_name=model_name)
    # sample it...
    return sm.sampling(**kwargs)


def get_cached_model_or_make_and_cache_it(model_code, model_name=None):
    filename = _get_filename_of_cached_model(model_code, model_name=model_name)
    sm = None
    for directory in stan_models_directory_paths:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            print(("Using cached StanModel found at %s" % filepath))
            sm = pickle.load(open(filepath, 'rb'))
            break
    if sm is None:
        if model_code is None:
            print("!!! No such model found")
        else:
            sm = pystan.StanModel(model_code=model_code)
            with open(filename, 'wb') as f:
                pickle.dump(sm, f)
    return sm


def _get_filename_of_cached_model(model_code, model_name=None):
    model_name = model_name or _get_model_name_from_code(model_code)
    return 'cached-{model_name}-{code_hash}.pkl'.format(
        model_name=model_name,
        code_hash=md5(model_code.encode('ascii')).hexdigest())


def _get_model_name_from_code(model_code):
    if model_code[:4] == '\n// ':
        try:
            model_name = re.search('\n// ([^\n]+)\n', model_code).group(1)
        except:
            model_name = 'model'
    else:
        model_name = 'model'
    return model_name



