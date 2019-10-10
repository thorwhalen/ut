__author__ = 'thor'


import copy
from itertools import combinations


class EdgeStats(object):

    def __init__(self):
        self._count = CountVal(0.0)
        self.a = KeyVal()
        self.ab = KeyVal()

    def count_data(self, item_iterator):
        self.__init__()
        for nodes in item_iterator:
            self._count.increment()
            for a in nodes:
                self.a.add(KeyVal({a: Val(1.0)}))
            for ab in combinations(nodes, 2):
                self.ab.add(KeyVal({ab[0]: KeyVal({ab[1]: Val(1.0)})}))
                self.ab.add(KeyVal({ab[1]: KeyVal({ab[0]: Val(1.0)})}))

    def counts_to_probs(self, alpha=0, beta=0):
        prior_num = Val(float(alpha))
        prior_denom = Val(float(alpha + beta))
        self.ab = (self.ab + prior_num) / (self.a + prior_denom)
        self.a = (self.a + prior_num) / (self._count + prior_denom)

    def __getitem__(self, item):
        return self.ab.v.get(item, None)


# default functions
def get_a_list_from_item_default(pair_set):
    return pair_set[0]


def get_b_list_from_item_default(pair_set):
    return pair_set[1]


class BipartiteEdgeCounts(object):
    """
    The class that manages the count data.
    """
#     _count
#     a_count
#     b_count
#     ab_count
#     ba_count

    def __init__(self, get_a_list_from_item=None, get_b_list_from_item=None):
        self._count = CountVal(0.0)
        self.a_count = KeyCount()
        self.b_count = KeyCount()
        self.ab_count = KeyCount()
        self.ba_count = KeyCount()
        self.get_a_list_from_item = get_a_list_from_item or get_a_list_from_item_default
        self.get_b_list_from_item = get_b_list_from_item or get_b_list_from_item_default

    def learn(self, item_iterator):
        self.__init__()
        for item in item_iterator:
            self._count.increment()
            a_list = self.get_a_list_from_item(item)
            b_list = self.get_b_list_from_item(item)
            for a in a_list:
                self.a_count.increment(a)
            for b in b_list:
                self.b_count.increment(b)
            for a in a_list:
                for b in b_list:
                    self.ab_count.add(KeyVal({a: KeyVal({b: Val(1.0)})}))
                    self.ba_count.add(KeyVal({b: KeyVal({a: Val(1.0)})}))


class Val(object):
    """
    The mother class of other Val classes.
    A Val should hold a value and be able to add and subtract from it.

    This mother class implements normal addition of floats, but should be overridden to
    implement other types of values such as multiplication, addition of vectors,
    merging of likelihoods etc.

    Most of the time, you'll only need to override the add() and the sub() methods.
    You may also want to override the default value. This value should act as the
    'unit' or 'neutral' value of the add operation (therefore the sub operation as well).
    For example, the unit value of multiplication (which will still be called "add") is 1.0.
    """
    v = 0.0

    def __init__(self, v):
        if isinstance(v, Val):
            self.v = copy.deepcopy(v.v)
        else:
            self.v = copy.deepcopy(v)

    def add(self, y):
        self.v = self.v + y.v

    def sub(self, y):
        self.v = self.v - y.v

    def mul(self, y):
        self.v = self.v * y.v

    def div(self, y):
        self.v = self.v / y.v

    def unwrapped(self):
        if hasattr(self.v, 'v'):
            return self.v.unwrapped()
        else:
            return self.v

    def map(self, fun):
        if hasattr(self.v, 'v'):
            self.v.map(fun)
        else:
            self.v = fun(self.v)

    def __add__(self, y):
        x = copy.deepcopy(self)
        x.add(y)
        return x

    def __sub__(self, y):
        x = copy.deepcopy(self)
        x.sub(y)
        return x

    def __mul__(self, y):
        x = copy.deepcopy(self)
        x.mul(y)
        return x

    def __div__(self, y):
        x = copy.deepcopy(self)
        x.div(y)
        return x

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return str(self.v)


class CountVal(Val):

    v = 0.0

    def __init__(self, v=0.0):
        super(CountVal, self).__init__(v)
        self.v = float(v)

    def increment(self):
        self.v += 1.0


class LHVal(Val):
    """
    An LHVal manages a binary likelihood.
    That is, it holds (as a single float) the binary likelihood distribution and allows one to
    merge two such distributions.
    """
    v = .5;  # where the value will be stored

    def __init__(self, v=.5):
        super(LHVal, self).__init__(v)
        self.v = float(v)

    def mul(self, y):
        self.v = (self.v * y.v) / (self.v * y.v + (1 - self.v) * (1 - y.v))

    def div(self, y):
        self.v = (self.v / y.v) / (self.v / y.v + (1 - self.v) / (1 - y.v))


class KeyVal(Val):
    """
    Here the type of the value is a dict (to implement a map).
    The addition of two dicts (therefore the add() method) v and w.

    The add(val) method will here be defined to be a sum-update of the (key,value)
    pairs of the
    Extends a map so that one can add and subtract dict pairs by adding or subtracting
     the (key-aligned) values
    """
    def __init__(self, v=None):
        if v is None:
            self.v = dict()
        else:
            super(KeyVal, self).__init__(v)

    def add(self, kv):
        try:
            if hasattr(kv.v, 'keys'):
                for k in list(kv.v.keys()):
                    if k in list(self.v.keys()):
                        self.v[k].add(kv.v[k])
                    else:
                        self.v[k] = kv.v[k]
            else:
                try:
                    for k in list(self.v.keys()):
                        self.v[k].v = self.v[k].v + kv.v
                except TypeError:
                    for k in list(self.v.keys()):
                        self.v[k].add(kv)
        except AttributeError:
            self.add(Val(kv))

    def sub(self, kv):
        try:
            if hasattr(kv.v, 'keys'):
                for k in list(kv.v.keys()):
                    if k in list(self.v.keys()):
                        self.v[k].sub(kv.v[k])
            else:
                try:
                    for k in list(self.v.keys()):
                        self.v[k].v = self.v[k].v - kv.v
                except TypeError:
                    for k in list(self.v.keys()):
                        self.v[k].sub(kv)
        except AttributeError:
            self.sub(Val(kv))

    def mul(self, kv):
        try:
            if hasattr(kv.v, 'keys'):
                for k in list(kv.v.keys()):
                    if k in list(self.v.keys()):
                        self.v[k].mul(kv.v[k])
                    else:
                        self.v[k] = kv.v[k]
            else:
                try:
                    for k in list(self.v.keys()):
                        self.v[k].v = self.v[k].v * kv.v
                except TypeError:
                    for k in list(self.v.keys()):
                        self.v[k].mul(kv)
        except AttributeError:
            self.mul(Val(kv))

    def div(self, kv):
        try:
            if hasattr(kv.v, 'keys'):
                for k in list(kv.v.keys()):
                    if k in list(self.v.keys()):
                        self.v[k].div(kv.v[k])
            else:
                try:
                    for k in list(self.v.keys()):
                        self.v[k].v = self.v[k].v / kv.v
                except TypeError:
                    for k in list(self.v.keys()):
                        self.v[k].div(kv)
        except AttributeError:
            self.div(Val(kv))

    def unwrapped(self):
        return {k: v.unwrapped() for k, v in self.v.items()}

    def map(self, fun):
        for v in self.v.values():
            v.map(fun)

    def keys(self):
        return list(self.v.keys())

    def iteritems(self):
        return iter(self.v.items())

    def __getitem__(self, item):
        return self.v.get(item, None)


class KeyCount(KeyVal):
#     v = dict()
#     init_val_constructor = None;
    """
    Extends a map so that one can add and subtract dict pairs by adding or subtracting the (key-aligned) values
    """
    def __init__(self, v=None):
        if v is None:
            self.v = dict()
        else:
            super(KeyCount, self).__init__(v)

    def increment(self, k):
        if k in self.v:
            self.v[k].add(Val(1.0))
        else:
            self.v[k] = Val(1.0)


# if __name__ == "__main__":
#     d = ut.daf.get.rand(nrows=9)
#     s = d['A'].iloc[0:5]
#     ss = d['B'].iloc[3:8]
#     t = s + ss
#     print t


