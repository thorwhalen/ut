__author__ = 'thor'

from decimal import Decimal
from numpy import *
from fractions import Fraction


def spread(start, end, count, mode=1):
    """spread(start, end, count [, mode]) -> generator

    Yield a sequence of evenly-spaced numbers between start and end.

    The range start...end is divided into count evenly-spaced (or as close to
    evenly-spaced as possible) intervals. The end-points of each interval are
    then yielded, optionally including or excluding start and end themselves.
    By default, start is included and end is excluded.

    For example, with start=0, end=2.1 and count=3, the range is divided into
    three intervals:

        (0.0)-----(0.7)-----(1.4)-----(2.1)

    resulting in:

        >>> list(spread(0.0, 2.1, 3))
        [0.0, 0.7, 1.4]

    Optional argument mode controls whether spread() includes the start and
    end values. mode must be an int. Bit zero of mode controls whether start
    is included (on) or excluded (off); bit one does the same for end. Hence:

        0 -> open interval (start and end both excluded)
        1 -> half-open (start included, end excluded)
        2 -> half open (start excluded, end included)
        3 -> closed (start and end both included)

    By default, mode=1 and only start is included in the output.

    (Note: depending on mode, the number of values returned can be count,
    count-1 or count+1.)
    """
    if not isinstance(mode, int):
        raise TypeError('mode must be an int')
    if count != int(count):
        raise ValueError('count must be an integer')
    if count <= 0:
        raise ValueError('count must be positive')
    if mode & 1:
        yield start
    width = Fraction(end-start)
    start = Fraction(start)
    for i in range(1, count):
        yield float(start + i*width/count)
    if mode & 2:
        yield end

def regulator(num_dec_places):
    if num_dec_places == 0:
        num_str = '0'
    else:
        num_str = '0.' + ('0' * (num_dec_places-1)) + '1'
    return num_str


def rounded_numeric_string(amount, num_dec_places=2):
    amount = str(amount)
    dec_amt = Decimal(amount.replace(',', ''))
    quantum = Decimal(regulator(num_dec_places))
    small_amount_for_better_rounding = Decimal(regulator(20))
    dec_amt += small_amount_for_better_rounding
    rounded = dec_amt.quantize(quantum)
    return str(rounded)


class DiscreteCoordLinearizer(object):
    def __init__(self, dims, order='C'):
        self.dims = dims
        self.order = order
        self.n_dims = len(dims)
        if self.order == 'C':
            self.base_key = flipud(self.dims_to_base_key(flipud(dims)))
        elif self.order == 'F':
            self.base_key = self.dims_to_base_key(dims)
        else:
            ValueError("Unknown order (should be 'C' or 'F')")

    @classmethod
    def dims_to_base_key(cls, dims):
        return concatenate([[1], cumprod(dims)[:-1]])

    def tuple_to_index(self, coord_tuple):
        """
        coord_tuple should be an iterable of (self.n_dims-)tuples
        """
        return sum(self.base_key * vstack(coord_tuple), axis=1)

    def index_to_tuple(self, idx):
        return vstack(unravel_index(idx, dims=self.dims, order=self.order)).T

    def n_pts(self):
        return prod(self.dims)


