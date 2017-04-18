from __future__ import division

from functools import update_wrapper
from functools import wraps
from inspect import getargspec, isfunction
from itertools import izip, ifilter, starmap

__author__ = 'thor'


def just_print_exceptions(func):
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
    return new_func


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


def autoassign(*names, **kwargs):
    """
    autoassign(function) -> method
    autoassign(*argnames) -> decorator
    autoassign(exclude=argnames) -> decorator

    allow a method to assign (some of) its arguments as attributes of
    'self' automatically.  E.g.

    >>> class Foo(object):
    ...     @autoassign
    ...     def __init__(self, foo, bar): pass
    ...
    >>> breakfast = Foo('spam', 'eggs')
    >>> breakfast.foo, breakfast.bar
    ('spam', 'eggs')

    To restrict autoassignment to 'bar' and 'baz', write:

        @autoassign('bar', 'baz')
        def method(self, foo, bar, baz): ...

    To prevent 'foo' and 'baz' from being autoassigned, use:

        @autoassign(exclude=('foo', 'baz'))
        def method(self, foo, bar, baz): ...
    """
    if kwargs:
        exclude, f = set(kwargs['exclude']), None
        sieve = lambda l: ifilter(lambda nv: nv[0] not in exclude, l)
    elif len(names) == 1 and isfunction(names[0]):
        f = names[0]
        sieve = lambda l: l
    else:
        names, f = set(names), None
        sieve = lambda l: ifilter(lambda nv: nv[0] in names, l)

    def decorator(f):
        fargnames, _, _, fdefaults = getargspec(f)
        # Remove self from fargnames and make sure fdefault is a tuple
        fargnames, fdefaults = fargnames[1:], fdefaults or ()
        defaults = list(sieve(izip(reversed(fargnames), reversed(fdefaults))))

        @wraps(f)
        def decorated(self, *args, **kwargs):
            assigned = dict(sieve(izip(fargnames, args)))
            assigned.update(sieve(kwargs.iteritems()))
            for _ in starmap(assigned.setdefault, defaults): pass
            self.__dict__.update(assigned)
            return f(self, *args, **kwargs)

        return decorated

    return f and decorator(f) or decorator