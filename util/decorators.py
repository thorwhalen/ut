from __future__ import division

from functools import update_wrapper
from functools import wraps
from inspect import getargspec, isfunction
from itertools import izip, ifilter, starmap
import inspect
import functools

__author__ = 'thor'

def autoargs(*include,**kwargs):
    """
    Automatically assign __init__ arguments.
    Courtesy of the answer of: https://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
    :param include: arguments to include
    :param kwargs: other kwargs to include (a 'exclude' key will have the effect of excluding the valued list from
        being auto-assigned.
    :return: a wrapped __init__ function.
    >>> from ut.util.decorators import autoargs
    ...
    ... class A(object):
    ...     @autoargs()
    ...     def __init__(self, foo, path, debug=False):
    ...         pass
    ... a = A('rhubarb', 'pie', debug=True)
    ... assert(a.foo == 'rhubarb')
    ... assert(a.path == 'pie')
    ... assert(a.debug == True)
    ...
    ... class B(object):
    ...     @autoargs()
    ...     def __init__(self, foo, path, debug=False, *args):
    ...         pass
    ... a = B('rhubarb', 'pie', True, 100, 101)
    ... assert(a.foo == 'rhubarb')
    ... assert(a.path == 'pie')
    ... assert(a.debug == True)
    ... assert(a.args == (100, 101))
    ...
    >>> class C(object):
    ...     @autoargs()
    ...     def __init__(self, foo, path, debug=False, *args, **kw):
    ...         pass
    ... a = C('rhubarb', 'pie', True, 100, 101, verbose=True)
    ... assert(a.foo == 'rhubarb')
    ... assert(a.path == 'pie')
    ... assert(a.debug == True)
    ... assert(a.verbose == True)
    ... assert(a.args == (100, 101))
    ...
    ...
    ... class C(object):
    ...     @autoargs('bar', 'baz', 'verbose')
    ...     def __init__(self, foo, bar, baz, verbose=False):
    ...         pass
    ... a = C('rhubarb', 'pie', 1)
    ... assert(a.bar == 'pie')
    ... assert(a.baz == 1)
    ... assert(a.verbose == False)
    ... try:
    ...     getattr(a, 'foo')
    ... except AttributeError:
    ...     print("Yep, that's what's expected!")
    ...
    ...
    ... class C(object):
    ...     @autoargs(exclude=('bar', 'baz', 'verbose'))
    ...     def __init__(self, foo, bar, baz, verbose=False):
    ...         pass
    ... a = C('rhubarb', 'pie', 1)
    ... assert(a.foo == 'rhubarb')
    ... try:
    ...     getattr(a, 'bar')
    ... except AttributeError:
    ...     print("Yep, that's what's expected!")
    ...
    Yep, that's what's expected!
    Yep, that's what's expected!
    """
    def _autoargs(func):
        attrs,varargs,varkw,defaults=inspect.getargspec(func)
        def sieve(attr):
            if kwargs and attr in kwargs['exclude']: return False
            if not include or attr in include: return True
            else: return False
        @functools.wraps(func)
        def wrapper(self,*args,**kwargs):
            # handle default values
            for attr,val in zip(reversed(attrs),reversed(defaults)):
                if sieve(attr): setattr(self, attr, val)
            # handle positional arguments
            positional_attrs=attrs[1:]
            for attr,val in zip(positional_attrs,args):
                if sieve(attr): setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args=args[len(positional_attrs):]
                if sieve(varargs): setattr(self, varargs, remaining_args)
            # handle varkw
            if kwargs:
                for attr,val in kwargs.iteritems():
                    if sieve(attr): setattr(self,attr,val)
            return func(self,*args,**kwargs)
        return wrapper
    return _autoargs


def lazyprop(fn):
    """
    Instead of having to implement the "if hasattr blah blah" code for lazy loading, just write the function that
    returns the value and decorate it with lazyprop! See example below.

    Taken from https://github.com/sorin/lazyprop.

    :param fn: The @property method (function) to implement lazy loading on
    :return: a decorated lazy loading property

    >>> class Test(object):
    ...     @lazyprop
    ...     def a(self):
    ...         print 'generating "a"'
    ...         return range(5)
    >>> t = Test()
    >>> t.__dict__
    {}
    >>> t.a
    generating "a"
    [0, 1, 2, 3, 4]
    >>> t.__dict__
    {'_lazy_a': [0, 1, 2, 3, 4]}
    >>> t.a
    [0, 1, 2, 3, 4]
    >>> del t.a
    >>> t.a
    generating "a"
    [0, 1, 2, 3, 4]
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    @_lazyprop.deleter
    def _lazyprop(self):
        if hasattr(self, attr_name):
            delattr(self, attr_name)

    @_lazyprop.setter
    def _lazyprop(self, value):
        setattr(self, attr_name, value)

    return _lazyprop


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


