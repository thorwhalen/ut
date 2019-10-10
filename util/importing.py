__author__ = 'thor'

from os import environ
from warnings import warn
import importlib
import os


def get_environment_variable(var, ignore=True):
    try:
        return environ[var]
    except KeyError:
        if ignore:
            warn(RuntimeWarning("You don't have the environment variable {}. Ignoring...".format(var)))
            return "You don't have the environment variable {}".format(var)
        else:
            raise RuntimeError


def module_import_from_string(import_path_string, params_file):
    if params_file.endswith('.py'):
        module_dir, params_filename = os.path.split(params_file)
        params_module, module_ext = os.path.splitext(params_filename)
    else:
        params_module = params_file
        params_file = params_module + '.py'

    print("")
    print(params_file)
    print(params_module)

    try:
        p = importlib.import_module(import_path_string + '.' + params_module)
    except ImportError as e:
        import imp
        import sys
        sys.path.append(os.path.dirname(os.path.expanduser(params_file)))
        p = importlib.import_module(params_module)

    return p