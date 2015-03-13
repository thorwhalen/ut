__author__ = 'thor'

import functional
import functools

def multi_compose(x):
    return functional.partial(functools.reduce, functional.compose)(x)


def return_none_on_error(func, *args, **kwargs):
    """
    utility to be able to "ignore" errors (by returning None) when calling a function
    """
    try:
        func(*args, **kwargs)
    except:
        return None


def pass_on_error(func, *args, **kwargs):
    """
    utility to be able to "ignore" error (by doing a 'pass') when calling a function
    """
    try:
        func(*args, **kwargs)
    except:
        pass


def handle_error(func, handle=lambda e: None, *args, **kwargs):
    """
    utility to be able to "handle" errors (by calling an input handle function on the error) when calling a function
    (default handle return None on error)
    """
    try:
        func(*args, **kwargs)
    except Exception as e:
        return handle(e)