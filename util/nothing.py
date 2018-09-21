from __future__ import division

from itertools import chain


def add_method(obj, meth, name=None, cname=None):
    if isinstance(meth, basestring):
        name = meth
        meth = getattr(obj, name)
    if name is None:
        name = meth.__name__

    base = type(obj)

    if cname is None:
        cname = "_".join((base.__name__, name, "add_method"))
    bases = (base.__bases__[1:]) + (base,)

    new_keys = set(dir(obj)) - set(chain(*[dir(b) for b in bases]))

    d = {a: getattr(obj, a) for a in new_keys}
    d[name] = meth
    return type(cname, bases, d)()


def transparent_method(self, x):
    return x


def get_universal_neutral_for_method(method_name):
    """
    A universal neutral element (synonyms: "identity element" and "unit element").
    A neutral element is with respect to a binary operation op(a, b).
    A neutral element e (for op) is one such that op(e, b) = b for any b.
    (Formally, we should also have op(b, e) = b, but we'll ignore that!)

    It is universal from the point of view of type, not method.
    Here, you specify what method (operation) you want it to be a neutral element for.
    Typically, you use this when you want to initialize a variable that will "accumulate/aggregate" objects one by one.
    For example, you say
        cumul = 0
    and then add elements to cumul
        cumul = cumul + x
    or
        cumul += x
    etc.

    Depending on what the type is, you need to initialize in different ways.
        cumul = 0   for ints
        cumul = []  for lists
        cumul = numpy.array(array_shape) for numpy arrays (and better know the dims of the arrays you'll be feeding!)
        cumul = "" for strings

    With the universal element, you don't need to initialize according to type. You just need to say:
        cumul = get_universal_neutral_for_method("__add__")
    and you'll have something that behaves as expected: Be transparent.

    Other examples of common methods where you want a neutral:  __mul__, update, etc.

    >>> nothing = get_universal_neutral_for_method('__add__')  # note, this is constructed ones, and used many times
    >>>
    >>> nothing + 2
    2
    >>> nothing + [1, 2, 3]
    [1, 2, 3]
    >>> print(nothing + "wassup world!")
    wassup world!
    >>>
    >>> # Useful as an alternative of using zeros(shape_of_your_array) to initialized array cumulators.
    >>> # No need to specify the array shape in advance!
    >>> import numpy as np
    >>> print(nothing + np.array([1, 2, 3, 4]))
    [1 2 3 4]
    >>> # and += works automatically!
    >>> t = nothing
    >>> t += np.array([1, 2, 3, 4])
    >>> print(t)
    [1 2 3 4]
    >>> t += np.array([10, 20, 30, 40])
    >>> print(t)
    [11 22 33 44]
    >>>
    >>> # Example with defaultdict (and multi type value cumulators
    >>> from collections import defaultdict
    >>> t = defaultdict(lambda: nothing)
    >>> t['foo'] += 2
    >>> t['bar'] += [1,2,3]
    >>> t['bar'] += [10,20]
    >>> t['is'] += np.array([1,2,3,4])
    >>> t['is'] += np.array([1,2,3,4])
    >>> t['overused'] += "hello"
    >>> t['overused'] += " "
    >>> t['overused'] += "world"
    >>> dict(t)
    {'overused': 'hello world', 'is': array([2, 4, 6, 8]), 'foo': 2, 'bar': [1, 2, 3, 10, 20]}
    >>>
    >>> # Works with other methods too!
    >>> nothing = get_universal_neutral_for_method('__mul__')
    >>> nothing * 314
    314
    >>> nothing * 'foobar'
    'foobar'
    """
    return add_method(object(), meth=transparent_method, name=method_name, cname=method_name + '_UniversalNeutral')


def get_universal_neutral_for_methods(method_names):
    """
    See get_universal_neutral_for_method(method_name).
    This is just the "multiple methods" version of that one.
    """
    nothing = object
    for method_name in method_names:
        nothing = add_method(nothing, meth=transparent_method, name=method_name, cname='UniversalNeutral')
    return nothing


class GreatUniversalNothing(object):
    """
    An object that is a neutral for any method (well, except those that count, like __add__, __mul__, etc.
    Needs to be finished to handle underscore methods
    """
    def __getattribute__(self, name):
        return lambda x: x



