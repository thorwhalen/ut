from __future__ import division

__author__ = 'thor'

from numpy import *
import numpy as np
from itertools import imap, product
from collections import defaultdict, Counter


def wp_iter_with_sliding_discrete_step(data_range,  # length of the interval we'll retrieve the windows from
                                       x_range=1,  # length of the y window
                                       y_range=1,  # length of the y window
                                       offset=0  # the offset from the end of the x window (0 when y window starts immediately after)
                                       ):
    """
    Returns a SLIDING DISCRETE STEP "window pair iterator".
    A "window pair iterator" is an iterator which yields a 4-tuple that are indices of two sliding windows.

    Usage:
    wp_iter_with_sliding_discrete_step(data_range,  # length of the interval we'll retrieve the windows from
                                           x_range=1,  # length of the y window
                                           y_range=1,  # length of the y window
                                           offset=0  # the offset from the end of the x window (0 when y window starts immediately after)
                                           ):

    Example:
    >>> from ut.iacc.seq.window_pairs import wp_iter_with_sliding_discrete_step
>>> from numpy import all, array
    >>> result = list(wp_iter_with_sliding_discrete_step(data_range=10, x_range=2, y_range=3, offset=1))
    >>> expected = [
    ...     array([0, 2, 3, 6]),
    ...     array([1, 3, 4, 7]),
    ...     array([2, 4, 5, 8]),
    ...     array([3, 5, 6, 9])]
    >>> assert all(array(result) == array(expected)), "result NOT as expected"
    """

    # input validation
    assert offset >= -x_range, "offset cannot be smaller than -x_spec['range'] (y-window can't start before x-window)"

    # compute the number of steps of the iteration
    n_steps = data_range - np.max([x_range, x_range + offset + y_range])

    if n_steps == 0:
        raise StopIteration()
    else:
        base_window_idx = array([0,
                                 x_range,
                                 x_range + offset,
                                 x_range + offset + y_range])
        for step in xrange(n_steps):
            yield base_window_idx + step
        raise StopIteration()


def _event_exists(arr):
    return int(any(arr))


def _columnwise_event_exists(df):
    return df.apply(_event_exists)


def extract_series(df,  # data to extract from
                         window_iterator=None,  # window iterator
                         x_extractor=_columnwise_event_exists,  # function to apply to the windowed df to get x
                         y_extractor=_columnwise_event_exists  # function to apply to the windowed df to get y
                        ):
    """

    """

    # combine the extractors
    def _extractor(window_idx):
        return (x_extractor(df.iloc[window_idx[0]:window_idx[1]]),
                y_extractor(df.iloc[window_idx[2]:window_idx[3]]))

    # get a default window_iterator if none given
    if window_iterator is None:
        window_iterator = wp_iter_with_sliding_discrete_step(data_range=len(df))

    # return the extractor iterator
    return imap(_extractor, window_iterator)


def agg_counts(pairs_of_series_iter):
    accum = defaultdict(Counter)

    for pair in pairs_of_series_iter:
        pair_iter = imap(lambda x: zip(*x), product(pair[0].to_dict().iteritems(), pair[1].to_dict().iteritems()))
        for x in pair_iter:
            accum[x[0]].update([x[1]])

    return accum