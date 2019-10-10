__author__ = 'thor'

import functools


def compose(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def filter_kwargs_to_func_arguments(func, kwargs):
    return dict([(k, v) for k, v in kwargs.items() if k in func.__code__.co_varnames])


def multi_compose(x):
    return functools.partial(functools.reduce, compose)(x)


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
