from functools import update_wrapper
from functools import wraps
from inspect import getargspec, isfunction
from itertools import starmap
import inspect
import functools

__author__ = 'thor'


def decorate_all_methods(decorator, exclude=(), include=()):
    def decorate(obj):
        if not include:
            inclusion_list = [x[0] for x in
                              [x for x in inspect.getmembers(obj) if not x[0].startswith('__') and callable(x[1])]]
        else:
            inclusion_list = include
        inclusion_list = list(set(inclusion_list).difference(exclude))
        inclusion_list = [x for x in inclusion_list if callable(getattr(obj, x))]
        inclusion_dict = {method_name: getattr(obj, method_name) for method_name in inclusion_list}
        for method_name, method in inclusion_dict.items():
            setattr(obj, method_name, decorator(method))
        return obj

    return decorate


def autoargs(*include, **kwargs):
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
        attrs, varargs, varkw, defaults = inspect.getargspec(func)

        def sieve(attr):
            if kwargs and attr in kwargs['exclude']: return False
            if not include or attr in include:
                return True
            else:
                return False

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # handle default values
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                if sieve(attr): setattr(self, attr, val)
            # handle positional arguments
            positional_attrs = attrs[1:]
            for attr, val in zip(positional_attrs, args):
                if sieve(attr): setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args = args[len(positional_attrs):]
                if sieve(varargs): setattr(self, varargs, remaining_args)
            # handle varkw
            if kwargs:
                for attr, val in kwargs.items():
                    if sieve(attr): setattr(self, attr, val)
            return func(self, *args, **kwargs)

        return wrapper

    return _autoargs


class lazyprop:
    """
    A descriptor implementation of lazyprop (cached property) from David Beazley's "Python Cookbook" book.
    It's
    >>> class Test:
    ...     def __init__(self, a):
    ...         self.a = a
    ...     @lazyprop
    ...     def len(self):
    ...         print('generating "len"')
    ...         return len(self.a)
    >>> t = Test([0, 1, 2, 3, 4])
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4]}
    >>> t.len
    generating "len"
    5
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4], 'len': 5}
    >>> t.len
    5
    >>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
    >>> t.a = [0, 1, 2]  # if we change a...
    >>> t.len  # ... we still get the old cached value of len
    5
    >>> del t.len  # if we delete the len prop
    >>> t.len  # ... then len being recomputed again
    generating "len"
    3
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


def lazy_immutable_prop(func):
    """ Slower version of lazyprop, but protects from assigning a value to the property """
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy


def old_lazyprop(fn):
    """
    Instead of having to implement the "if hasattr blah blah" code for lazy loading, just write the function that
    returns the value and decorate it with lazyprop! See example below.

    Taken from https://github.com/sorin/lazyprop.

    :param fn: The @property method (function) to implement lazy loading on
    :return: a decorated lazy loading property

    >>> class Test(object):
    ...     @old_lazyprop
    ...     def a(self):
    ...         print('generating "a"')
    ...         return list(range(5))
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
        _total_names = inFunction.__code__.co_varnames[1:inFunction.__code__.co_argcount]
        # Get all of the values
        _values = args[1:]
        # Get only the names that don't belong to kwargs
        _names = [n for n in _total_names if n not in kwargs]

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
        sieve = lambda l: filter(lambda nv: nv[0] not in exclude, l)
    elif len(names) == 1 and isfunction(names[0]):
        f = names[0]
        sieve = lambda l: l
    else:
        names, f = set(names), None
        sieve = lambda l: filter(lambda nv: nv[0] in names, l)

    def decorator(f):
        fargnames, _, _, fdefaults = getargspec(f)
        # Remove self from fargnames and make sure fdefault is a tuple
        fargnames, fdefaults = fargnames[1:], fdefaults or ()
        defaults = list(sieve(zip(reversed(fargnames), reversed(fdefaults))))

        @wraps(f)
        def decorated(self, *args, **kwargs):
            assigned = dict(sieve(zip(fargnames, args)))
            assigned.update(sieve(iter(kwargs.items())))
            for _ in starmap(assigned.setdefault, defaults): pass
            self.__dict__.update(assigned)
            return f(self, *args, **kwargs)

        return decorated

    return f and decorator(f) or decorator
