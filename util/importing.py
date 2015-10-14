__author__ = 'thor'

from os import environ
from warnings import warn


def get_environment_variable(var, ignore=True):
    try:
        return environ[var]
    except KeyError:
        if ignore:
            warn(RuntimeWarning("You don't have the environment variable {}. Ignoring...".format(var)))
            return "You don't have the environment variable {}".format(var)
        else:
            raise(RuntimeError("You don't have the environment variable {}.".format(var)))


