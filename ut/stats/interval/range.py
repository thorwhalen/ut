"""Interval arithmetic"""

from numpy import inf, random, array, ravel, amin, amax, ndim, vectorize
from ut.ml.util.feature_analysis import plot_feat_ranges


class Range(object):
    """
    Class implementing simple interval arithmetic
    """

    def __init__(self, lower, upper):
        """Create interval [m,M]"""
        if lower <= upper:
            self.m = float(lower)
            self.M = float(upper)
        else:
            raise ValueError(
                'lower limit %s must be smaller than upper limit %s' % (lower, upper)
            )

    @staticmethod
    def n2i(n):
        """
        Transforms a number or interval into interval.
        """
        if isinstance(n, (int, float)):
            return Range(n, n)
        elif isinstance(n, Range):
            return n
        else:
            raise TypeError('Given %s %s must be number or ' 'interval' % (n, type(n)))

    def min(self):
        return self.m

    def max(self):
        return self.m

    def _limits(self, other):
        other = Range.n2i(other)
        return self.m, self.M, other.m, other.M

    def __len__(self):
        return self.M - self.m

    def __contains__(self, other):
        """
        True if self range completely contains other range.
        """
        return self.m <= other.m and self.M >= other.M

    def __add__(self, other):
        a, b, c, d = self._limits(other)
        return Range(a + c, b + d)

    def __sub__(self, other):
        a, b, c, d = self._limits(other)
        return Range(a - d, b - c)

    def __mul__(self, other):
        a, b, c, d = self._limits(other)
        return Range(min(a * c, a * d, b * c, b * d), max(a * c, a * d, b * c, b * d))

    def __truediv__(self, other):
        a, b, c, d = self._limits(other)
        # # [c,d] cannot contain zero:
        # if c * d <= 0:
        # raise ValueError \
        #     ('Interval %s cannot be inverted because ' \
        #      'it contains zero')
        return Range(min(a / c, a / d, b / c, b / d), max(a / c, a / d, b / c, b / d))

    def __div__(self, other):
        return self.__truediv__(other)

    def __radd__(self, other):
        other = Range.n2i(other)
        return other + self

    def __rsub__(self, other):
        other = Range.n2i(other)
        return other - self

    def __rmul__(self, other):
        other = Range.n2i(other)
        return other * self

    def __rdiv__(self, other):
        other = Range.n2i(other)
        return other / self

    def __pow__(self, exponent):
        if isinstance(exponent, int):
            p = 1
            if exponent > 0:
                for i in range(exponent):
                    p = p * self
            elif exponent < 0:
                for i in range(-exponent):
                    p = p * self
                p = 1 / p
            else:
                p = Range(1, 1)
            return p
        else:
            raise TypeError('Exponent must be integer')

    def __eq__(self, other):
        return self.m == other.m and self.M == other.M

    def __neq__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.M < other.m

    def __le__(self, other):
        return self.M <= other.m

    def __gt__(self, other):
        return self.m > other.M

    def __ge__(self, other):
        return self.m >= other.M

    def abs(self):
        abs_m = abs(self.m)
        abs_M = abs(self.M)
        if abs_m < abs_M:
            return Range(lower=abs_m, upper=abs_M)
        else:
            return Range(lower=abs_M, upper=abs_m)

    # def __float__(self):
    #     return 0.5 * (self.m + self.M)

    def center(self):
        return 0.5 * (self.m + self.M)

    def width_in_percent(self):
        """
        Return the width of the interval as percentage around the mean.
        """
        I = self.center()
        w2 = I - self.m
        p2 = w2 / I * 100
        return 2 * p2

    def tolist(self):
        return [self.m, self.M]

    def __str__(self):
        return '[%g, %g]' % (self.m, self.M)

    def __repr__(self):
        return '%s(%g, %g)' % (self.__class__.__name__, self.m, self.M)

    def min_dist_to_interval(self, c):
        """
        Return closest point to c.
        """
        min_int = self.m
        max_int = self.M
        if min_int <= c <= max_int:
            closest = c
        elif c < min_int:
            closest = min_int
        else:
            closest = max_int

        return closest

    def max_dist_to_interval(self, c):
        """
        Returns the furthest point to c.
        """
        min_int = self.m
        max_int = self.M
        if min_int <= c <= max_int:
            if abs(min_int - c) > abs(max_int - c):
                farthest = min_int
            else:
                farthest = max_int
        elif c < min_int:
            farthest = max_int
        else:
            farthest = min_int

        return farthest

    def is_healthy(self):
        return self.m <= self.M

    def __iter__(self):
        yield self.m
        yield self.M

    @classmethod
    def plot(cls, array_of_ranges, **kwargs):
        plot_feat_ranges(list(map(tuple, ravel(array_of_ranges))), **kwargs)

    @classmethod
    def from_array(cls, a, axis=None):
        if ndim(a) == 1 or axis is None:
            return cls(amin(a), amax(a))
        else:
            return array(
                list(map(cls, ravel(amin(a, axis=axis)), ravel(amax(a, axis=axis))))
            )

    @classmethod
    def lower_bound_array(cls, range_array):
        return vectorize(lambda x: x.m)(range_array)

    @classmethod
    def upper_bound_array(cls, range_array):
        return vectorize(lambda x: x.M)(range_array)

    @classmethod
    def amin(cls, range_array, axis=None):
        return amin(cls.lower_bound_array(range_array), axis=axis)

    @classmethod
    def amax(cls, range_array, axis=None):
        return amax(cls.upper_bound_array(range_array), axis=axis)

    @classmethod
    def rand(cls, m=0, M=10, kind=int):
        """
        Util to quickly get a (single) Range.
        To get an array of Ranges, use array(map(lambda x: Range.rand(), xrange(NUM_OF_RANGES_YOU_WANT)))
        """
        if kind == float:
            mm, MM = m + array(sorted(random.rand(2) * (M - m)))
        else:
            mm, MM = sorted(random.randint(0, M, size=2))
        return cls(lower=mm, upper=MM)
