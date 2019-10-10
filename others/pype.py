# Author: Arvind Narayanan <randomwalker@gmail.com>
# http://randomwalker.info/pype
# svn://randomwalker.info/pype
# License: GPL v2 (http://www.gnu.org/licenses/gpl-2.0.html)
# see: http://arvindn.livejournal.com/68137.html
#
# thorwhalen added a __call_ to Pype

import sys, string, itertools, operator, copy, re
from functools import reduce


class Pype:
    def __init__(self, func, *args, **kwargs):
        """the *args and **kwargs will get passed to func"""
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def fix(self, *args, **kwargs):
        """use this function to supply more args at any time"""
        return type(self)(self.func, *(self.args + args), **dict(list(self.kwargs.items()) + list(kwargs.items())))

    def __or__(self, rhs):
        """combines two Pypes into a Chain"""
        return Chain(self, rhs)

    def __call__(self, lhs):
        return self.__ror__(lhs)


class Map(Pype):
    def __ror__(self, lhs):
        return map(lambda x: self.func(x, *self.args, **self.kwargs), lhs)


class Reduce(Pype):
    def __ror__(self, lhs):
        return reduce(self.func, lhs, *self.args)


class Sink(Pype):
    def __ror__(self, lhs):
        return self.func(lhs, *self.args, **self.kwargs)


class Filter(Pype):
    def __ror__(self, lhs):
        return filter(lambda x: self.func(x, *self.args, **self.kwargs), lhs)


class Chain(Pype):
    def __init__(self, lpype, rpype):
        self.lpype = lpype
        self.rpype = rpype

    def __ror__(self, lhs):
        return lhs | self.lpype | self.rpype


def pypeThunk(pypeclass, func):
    return lambda *args, **kwargs: pypeclass(lambda line: func(line, *args, **kwargs))


pFilter = lambda f: Sink(lambda x: filter(f, x))
pDict = Sink(dict)
pSplit = pypeThunk(Map, string.split)
pReverse = Sink(lambda x: list(x)[::-1])
pSum = Sink(sum)
pWhile = lambda pred: Sink(lambda func: itertools.takewhile(pred, func))
pJoin = lambda sep: Sink(sep.join)
pLen = Sink(lambda x: len(list(x)))
pSet = Sink(set)
pStrip = Map(string.strip)
pSort = pypeThunk(Sink, sorted)
pValues = Sink(lambda dic: iter(dic.values()))
pItems = Sink(lambda dic: iter(dic.items()))
pList = Sink(list)
pHead = lambda count: Sink(lambda iter: itertools.islice(iter, count))
pTail = lambda n: pReverse | pHead(n) | pReverse  # FIXME: implement w/o memory
pLower = Map(string.lower)
pUpper = Map(string.upper)


def _grep(line, word, invert=False, matchcase=True):
    if matchcase:
        line = line.lower()
        word = word.lower()
    found = re.compile(word).search(line) is not None
    return invert ^ found


def _fgrep(line, words, invert=False, matchcase=True):
    if matchcase:
        line = line.lower()
        words = list(map(string.lower, words))
    found = reduce(operator.or_, (line.find(word) >= 0 for word in words))
    return invert ^ found


pGrep = pypeThunk(Filter, _grep)
pFgrep = pypeThunk(Filter, _fgrep)


def _first(iter):
    l = list(iter)
    if len(l) == 0: return None
    return l[0]


pFirst = Sink(_first)


def getHist(values, mincount=0):
    hist = {}

    def add(v):
        try:
            hist[v] += 1
        except:
            hist[v] = 1

    for v in values:
        if isinstance(v, list):
            list(map(add, v))
        else:
            add(v)
    for key in copy.copy(hist):
        if hist[key] < mincount:
            del hist[key]
    return hist


pHist = Sink(getHist)


def _cut(lines, fields=0, delim=None, odelim=" ", suppress=False):
    """a more sensible version of the GNU cut utility: cuts on whitespace by default"""
    for line in lines:
        words = line.split(delim)
        try:
            yield odelim.join([words[f] for f in fields]) if isinstance(fields, list) else words[fields]
        except IndexError:
            if not suppress: yield ""


pCut = pypeThunk(Sink, _cut)


def _uniq(values):
    """removes duplicates, just like the UNIX command uniq"""
    first = True
    prev = None
    for v in values:
        sameasprev = v == prev
        first = False
        prev = v
        if first or not sameasprev: yield v


pUniq = Sink(_uniq)


def _writeToFile(values, filename=None, file=None, delim="\n"):
    out = open(filename, "w") if filename else file
    for v in values:
        out.write("%s%s" % (v, delim))
    if filename: out.close()


pWrite = pypeThunk(Sink, _writeToFile)
pPrint = Sink(_writeToFile, file=sys.stdout)
pError = Sink(_writeToFile, file=sys.stderr)

"""export all pypes, and the Pype subclasses"""
__all__ = [w for w in dir() if w[0] == 'p'] + "Map Reduce Sink Filter".split()