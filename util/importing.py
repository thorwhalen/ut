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


def module_import_from_string(import_path_string, params_file, verbose=False):
    if params_file.endswith('.py'):
        module_dir, params_filename = os.path.split(params_file)
        params_module, module_ext = os.path.splitext(params_filename)
    else:
        params_module = params_file
        params_file = params_module + '.py'

    if verbose:
        print(f"params_file={params_file}, params_module={params_module}")

    try:
        p = importlib.import_module(import_path_string + '.' + params_module)
    except ImportError as e:
        import imp
        import sys
        sys.path.append(os.path.dirname(os.path.expanduser(params_file)))
        p = importlib.import_module(params_module)

    return p


def import_from_dot_string(dot_string):
    """
    Import an object from a dot string path to the object.
    :param dot_string: The dot-path to the module
    :return: The desired object

    >>> import os
    >>> module = type(os)
    >>> # can import modules
    >>> assert import_from_dot_string('os') == os
    >>> assert import_from_dot_string('os.path') == os.path
    >>> assert isinstance(os.path, module)
    >>> # can import strings
    >>> assert import_from_dot_string('os.path.sep') == os.path.sep
    >>> assert isinstance(os.path.sep, str)
    >>> # can import callables
    >>> assert import_from_dot_string('os.path.join') == os.path.join
    >>> assert callable(os.path.join)
    >>> # basically, can import objects
    """
    dot_list = dot_string.split('.')
    if len(dot_list) > 1:
        prefix = '.'.join(dot_list[:-1])
        suffix = dot_list[-1]
        module = importlib.import_module(prefix)
        obj = getattr(module, suffix)
        return obj
    else:
        return importlib.import_module(dot_string)
