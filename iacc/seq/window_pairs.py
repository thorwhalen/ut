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


# Indices into wp_key (for wp_iter_slide_to_next_event function)
PAST_HORIZON_TIME = 0
PAST_HORIZON_IDX = 1
PRESENT_TIME = 2
PRESENT_IDX = 3
FUTURE_HORIZON_TIME = 4
FUTURE_HORIZON_IDX = 5


def wp_iter_slide_to_next_event(timestamp_seq, past_range, future_range=None, min_timestamp=None, max_timestamp=None):
    """
    A window pairs iterator that slides the (past,future) windows (through a sequence of events whose timestamps
    are given by the input timestamp_seq) capturing the times when an event enters or leaves the windows
    (thus allowing to extract all pairs of possible states in cases where states are completely defined by
    the subsequence of events in the window, not their actual timestamps).
     More precisely, the (past,future) windows are indexed by the triple (past_horizon, present, future_horizon) where
        past_horizon is the beginning of past
        present is both the end of past and the beginning of future
        future_horizon is the end of future

    past_horizon      past      present         future      future_horizon
        [--------------------------[------------------------------[

     The first XY window is set so that past_horizon is at min_timestamp (defaulted to the lowest date in df.
     Then, at any point, the next window is chosen such that either of these conditions hold:
        (1) Some event leaves past (meaning event_date < past_horizon
        (2) Some event enters past (equivalent to leaving future) (meaning event_date < present)
        (3) Some event enters future (meaning event_date < future_horizon)
        (4) future_horizon reaches max_timestamp

    min_timestamp and max_timestamp are defaulted to the min and max date of df.
     The reason for being able to specify min_timestamp and max_timestamp is that the data of df might come from a
     set of event sequences that have a specific observation range, and we'd like to take into account the
     "no event" cases.

    The iterator yields 4-tuples (past_horizon_idx, present_idx, future_horizon_idx, duration) where the first three
    elements are indices of timestamp_seq and duration is the amount of time between the window pair associated to
    this 4-tuple and the next window pair.

    Note: With abuse of notation,
            past_horizon <= past < present
        and
            present <= future < future_horizon
    """
    timestamp_seq = sorted(timestamp_seq)
    timestamp_seq = timestamp_seq - timestamp_seq[0]
    if future_range is None:
        past_range = future_range
    if min_timestamp is None:
        min_timestamp = timestamp_seq[0]
    if max_timestamp is None:
        max_timestamp = timestamp_seq[-1]

    def _first_index_greater_or_equal_to(thresh, start_idx):
        for i, x in enumerate(timestamp_seq[start_idx:], start_idx):
            if x >= thresh:
                return i
        return None

    def _next_window(window_keys):
        pass

    # Initialize wp_key
    wp_key = zeros(6)
    wp_key[PAST_HORIZON_TIME] = timestamp_seq[0]
    wp_key[PAST_HORIZON_IDX] = 0
    wp_key[PRESENT_TIME] = wp_key[PAST_HORIZON_TIME] + past_range
    wp_key[PRESENT_IDX] = _first_index_greater_or_equal_to(wp_key[PRESENT_TIME], wp_key[PAST_HORIZON_IDX])
    wp_key[FUTURE_HORIZON_TIME] = wp_key[PRESENT_TIME] + future_range
    wp_key[FUTURE_HORIZON_IDX] = _first_index_greater_or_equal_to(wp_key[FUTURE_HORIZON_TIME], wp_key[PRESENT_IDX])

    past_horizon_time = timestamp_seq[0]
    past_horizon_idx = 0

    present_time = past_horizon_time + past_range
    present_idx = _first_index_greater_or_equal_to(present_time, past_horizon_idx)

    future_horizon_time = present_time + future_range
    future_horizon_idx = _first_index_greater_or_equal_to(future_horizon_time, present_idx)

    while future_horizon_idx is not None:
        pass



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