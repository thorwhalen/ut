from __future__ import division

from functools import update_wrapper

__author__ = 'thor'


def inject_param_initialization(inFunction):
    """
    This function allows to reduce code for initialization of parameters of a method through the @-notation
    You need to call this function before the method in this way: @inject_arguments
    """

    def outFunction(*args, **kwargs):
        _self = args[0]
        _self.__dict__.update(kwargs)
        # Get all of argument's names of the inFunction
        _total_names = inFunction.func_code.co_varnames[1:inFunction.func_code.co_argcount]
        # Get all of the values
        _values = args[1:]
        # Get only the names that don't belong to kwargs
        _names = [n for n in _total_names if not kwargs.has_key(n)]

        # Match names with values and update __dict__
        d = {}
        for n, v in zip(_names, _values):
            d[n] = v
        _self.__dict__.update(d)
        inFunction(*args, **kwargs)

    return update_wrapper(outFunction, inFunction)


