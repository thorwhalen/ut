import itertools
import re


class PVar:
    p = re.compile(r'^(.+)-(\d+)$|^(.+)$')

    def __init__(self, var: str, i: int = 0):
        self.var = var
        self.i = i

    def _tuple_for_ordering(self):
        return (self.i, self.var)

    def __eq__(self, other):
        return self._tuple_for_ordering().__eq__(other._tuple_for_ordering())
        # return self.var == other.var and self.i == other.i

    def __lt__(self, other):
        return self._tuple_for_ordering().__lt__(other._tuple_for_ordering())

    def __le__(self, other):
        return self._tuple_for_ordering().__le__(other._tuple_for_ordering())

    def __gt__(self, other):
        return self._tuple_for_ordering().__gt__(other._tuple_for_ordering())

    def __ge__(self, other):
        return self._tuple_for_ordering().__ge__(other._tuple_for_ordering())

    @classmethod
    def from_(cls, x):
        if isinstance(x, cls):
            return cls(x.var, x.i)  # make a copy of the Key object
        elif isinstance(x, tuple):
            return cls(*x)  # assume it's a (var, i) tuple
        elif isinstance(x, str):
            return cls.from_str(x)
        else:
            return cls(*x.__iter__())

    def __iter__(self):
        return self.var, self.i

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.var}', {self.i})"

    def __hash__(self):
        return hash(self.__repr__())

    def __str__(self):
        if self.i == 0:
            return f'{self.var}'
        else:
            return f'{self.var}{self.i}'

    @classmethod
    def from_str(cls, s):
        s = s.strip()
        m = cls.p.match(s)
        g = m.groups()
        if g[-1] is None:
            return cls(var=g[0], i=-int(g[1]))
        else:
            return cls(var=g[-1])

    def __getitem__(self, i):
        return PVar(self.var, self.i + i)

    def __add__(self, i):
        return PVar(self.var, self.i + i)

    def __sub__(self, i):
        return PVar(self.var, self.i - i)

    def __mul__(self, other):
        return VarSet([self, other])


class VarSet:
    def __init__(self, *varset):
        if len(varset) == 1 and isinstance(varset[0], (tuple, list, VarSet)):
            varset = varset[0]
        varset = list(map(PVar.from_, varset))
        self.varset = sorted(varset)
        self.min_abs_i = abs(
            min(x.i for x in self)
        )  # TODO: Not enough: Need to check on upper bound of sliding win

    @property
    def varset_strs(self):
        return list(map(str, self.varset))

    def __mul__(self, other):
        if isinstance(other, PVar):
            return VarSet(self.varset + [other])
        else:
            return VarSet(self.varset + other)

    def __eq__(self, other):
        if len(self.varset) != len(other.varset):
            return False
        else:
            for k, kk in zip(self.varset, other.varset):
                if k != kk:
                    return False
            return True

    def __iter__(self):
        return iter(self.varset)

    def __repr__(self):
        s = ', '.join(map(lambda x: f"'{x}'", self.varset))
        return f'{self.__class__.__name__}({s})'

    def __str__(self):
        return '(' + ', '.join(map(str, self.varset)) + ')'

    def __hash__(self):
        return hash(self.__repr__())

    def __getitem__(self, i):
        varset = [k[i] for k in self.varset]
        return VarSet(varset)


class VarSetFactory:
    @staticmethod
    def single_dim_markovs(varnames):
        return map(lambda v: VarSet([(v, -1), (v, 0)]), varnames)

    @staticmethod
    def pairs(varnames):
        return map(VarSet, itertools.combinations(map(PVar, varnames), 2))

    @staticmethod
    def tuples(varnames, tuple_size: int = 2):
        return map(VarSet, itertools.combinations(map(PVar, varnames), tuple_size))

    @staticmethod
    def markov_pairs(varnames):
        return map(
            lambda v: VarSet(PVar(v[0], -1), PVar(v[1], 0)),
            itertools.product(varnames, varnames),
        )

    @staticmethod
    def from_str(s):
        return VarSet(*map(PVar.from_str, list(s[1:-1].split(','))))

    # @classmethod
    # def pairs(cls, vars):
    #     return list(itertools.combinations(vars, 2))


def extract_kps(df, kps):
    # keep only elements of kps that have columns for them
    cols = set(df.columns)
    _kps = list()
    for k in kps:
        if k.var in cols:
            _kps.append(k)

    for i in range(len(df)):
        if i >= kps.min_abs_i:
            yield tuple(df[k.var].iloc[i + k.i] for k in _kps)


class DfData:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, k):
        if isinstance(k, PVar):
            return self.df[k.var].iloc[k.i]
        elif isinstance(k, VarSet):
            return tuple(map(self.__getitem__, k))

    def extract_with_key_pattern_sets(self, kps_list):
        for i in range(len(self.df)):
            for kps in kps_list:
                if i >= kps.min_abs_i:
                    yield kps, self[kps[i]]

    def extract_kps(self, kps):
        for i in range(len(self.df)):
            if i >= kps.min_abs_i:
                yield self[kps[i]]
