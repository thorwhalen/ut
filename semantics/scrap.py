__author__ = 'thor'

from ut.util.decorators import autoargs

class A(object):
    @autoargs()
    def __init__(self, foo, path, debug=False):
        pass
a = A('rhubarb', 'pie', debug=True)
assert(a.foo == 'rhubarb')
assert(a.path == 'pie')
assert(a.debug == True)

class B(object):
    @autoargs()
    def __init__(self, foo, path, debug=False, *args):
        pass
a = B('rhubarb', 'pie', True, 100, 101)
assert(a.foo == 'rhubarb')
assert(a.path == 'pie')
assert(a.debug == True)
assert(a.args == (100, 101))

class C(object):
    @autoargs()
    def __init__(self, foo, path, debug=False, *args, **kw):
        pass
a = C('rhubarb', 'pie', True, 100, 101, verbose=True)
assert(a.foo == 'rhubarb')
assert(a.path == 'pie')
assert(a.debug == True)
assert(a.verbose == True)
assert(a.args == (100, 101))


class C(object):
    @autoargs('bar', 'baz', 'verbose')
    def __init__(self, foo, bar, baz, verbose=False):
        pass
a = C('rhubarb', 'pie', 1)
assert(a.bar == 'pie')
assert(a.baz == 1)
assert(a.verbose == False)
try:
    getattr(a, 'foo')
except AttributeError:
    print("Yep, that's what's expected!")


class C(object):
    @autoargs(exclude=('bar', 'baz', 'verbose'))
    def __init__(self, foo, bar, baz, verbose=False):
        pass
a = C('rhubarb', 'pie', 1)
assert(a.foo == 'rhubarb')
try:
    getattr(a, 'bar')
except AttributeError:
    print("Yep, that's what's expected!")
