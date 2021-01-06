"""Window pairs of sequences"""

__author__ = 'thor'

from numpy import *
import numpy as np
from itertools import product
from collections import defaultdict, Counter

DEBUG_LEVEL = 0

def wp_iter_with_sliding_discrete_step(data_range,  # length of the interval we'll retrieve the windows from
                                       x_range=1,  # length of the x window
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
        for step in range(n_steps):
            yield base_window_idx + step
        raise StopIteration()


# Indices into wp_key (for wp_iter_slide_to_next_event function)
# PAST_HORIZON_TIME = 0
# PAST_HORIZON_IDX = 1
# PRESENT_TIME = 2
# PRESENT_IDX = 3
# FUTURE_HORIZON_TIME = 4
# FUTURE_HORIZON_IDX = 5

def slow_past_present_future_idx_and_duration_iter(timestamp_seq,
                                              past_range,
                                              future_range=None,
                                              timestamps_are_sorted=True):
    if not timestamps_are_sorted:
        timestamp_seq = sorted(timestamp_seq)
    timestamp_seq = array(timestamp_seq)
    if future_range is None:
        future_range = past_range
    _past_ts = timestamp_seq[0]
    _present_ts = _past_ts + past_range
    _future_ts = _present_ts + future_range
    max_timestamp = timestamp_seq[-1]
    while _future_ts < max_timestamp:
        gte_past_lidx = timestamp_seq > _past_ts
        gte_present_lidx = timestamp_seq > _present_ts
        gte_future_lidx = timestamp_seq > _future_ts

        # figure out next shift
        _shift = timestamp_seq[gte_past_lidx][0] - _past_ts
        next_present_distance = timestamp_seq[gte_present_lidx][0] - _present_ts
        if next_present_distance < _shift:
            _shift = next_present_distance
        next_future_distance = timestamp_seq[gte_future_lidx][0] - _future_ts
        if next_future_distance < _shift:
            _shift = next_future_distance
        # yield the indices triple, duration, and _present_ts
        yield ((where(timestamp_seq > _past_ts)[0][0],
               where(timestamp_seq <= _present_ts)[0][-1],
               where(timestamp_seq <= _future_ts)[0][-1]),
               _shift,
               _present_ts)

        # shift past, present and future
        _past_ts += _shift
        _present_ts += _shift
        _future_ts += _shift


def _idx_and_duration(timestamp_seq, timestamps_are_sorted=False):
    """
    returns a pair (idx, dur) where
        * idx is the first idx (into timestamp_seq) of every subsequence of equal values, and
        * dur is the duration of this subsequence (i.e. the time until the first different timestamp value
    Note: It is assumed (but not verified, for speed) that the sequence timestamp_seq of timestamps are ORDERED.
    """
    if not timestamps_are_sorted:
        timestamp_seq = sorted(timestamp_seq)
    idx = defaultdict(list)
    idx[0] = [0]
    idx_i = 0
    dur = []
    unik_timestamps = [timestamp_seq[0]]
    cumulating_zero_durs = False
    for i in range(1, len(timestamp_seq)):
        _dur = timestamp_seq[i] - timestamp_seq[i-1]
        if _dur == 0:
            if not cumulating_zero_durs:
                cumulating_zero_durs = True
            idx[idx_i].append(i)
        else:
            dur.append(_dur)
            idx_i += 1
            idx[idx_i].append(i)
            unik_timestamps.append(timestamp_seq[i])
            cumulating_zero_durs = False

    return dict(idx), dur, array(unik_timestamps)


_past = 0
_present = 1
_future = 2
_idx = 0
_time_to_next = 1


def past_present_future_idx_and_duration_iter(timestamp_seq,
                                              past_range,
                                              future_range=None,
                                              timestamps_are_sorted=False):
    """
    A window pairs iterator that slides the (past,future) windows (through a sequence of events whose timestamps
    are given by the input timestamp_seq) capturing the times when an event enters or leaves the windows
    (thus allowing to extract all pairs of possible states in cases where states are completely defined by
    the subsequence of events in the window, not their actual timestamps).
     More precisely, the (past,future) windows are indexed by the triple (past_horizon, present, future_horizon) where
        past_horizon is the beginning of past (inclusive)
        present is both the end of past (non inclusive) and the beginning of future (inclusive)
        future_horizon is the end of future (non inclusive)

    past_horizon      past      present         future      future_horizon
        [--------------------------[------------------------------[

     The first XY window is set so that past_horizon is at min_timestamp (defaulted to the lowest date in df.
     Then, at any point, the next window is chosen such that either of these conditions hold:
        (1) Some event leaves past (meaning event_date < past_horizon
        (2) Some event enters past (equivalent to leaving future) (meaning event_date < present)
        (3) Some event enters future (meaning event_date < future_horizon)
        (4) future_horizon reaches max_timestamp

    min_timestamp and max_timestamp are defaulted to the min and max date of timestamp_seq.
     The reason for being able to specify min_timestamp and max_timestamp is that the data of df might come from a
     set of event sequences that have a specific observation range, and we'd like to take into account the
     "no event" cases.

    The iterator yields a pair (ppf_idx, duration) where ppf = (past_horizon_idx, present_idx, future_horizon_idx)
    are indices of timestamp_seq and duration is the amount of time between the window pair associated to
    this pair and the next window pair.

    Note: With abuse of notation,
            past_horizon <= past < present <= future < future_horizon

    The output_present_timestamp==True option yields triples (ppf_idx, duration, present_timestamp) providing
    additionally the present_timestamp information (which is the timestamp of the present point that is used by the
    double window.

    Implementation details: The generator maintains a state, which is a 3x2 matrix where rows index past/present/future,
    and columns index _idx (an index to the last unique values of timestamps_seq) and _time_to_next (which indicates
    how far (in time units) the past/present/future point is to the next data point (a timestamp). The generator also
    maintains time_argmin which is the row index of the smallest _time_to_next.
        The algorithm iteratively updates the state and _time_to_next in order to get the tuples it generates.

    Below are two (doctest) examples:

    >>> from numpy import *
    >>> from matplotlib.cbook import flatten
    >>> timestamp_seq = [0, 5, 6, 15]
    >>> result = list(past_present_future_idx_and_duration_iter(timestamp_seq, 4))
    >>> expected = array([\
            ((1, 0, 2), 1, 4), \
            ((1, 1, 2), 1, 5), \
            ((1, 2, 2), 3, 6), \
            ((2, 2, 2), 1, 9), \
            ((3, 2, 2), 1, 10)])
    >>> all(array(list(flatten(result))) == array(list(flatten(expected))))
    True
    >>> timestamp_seq = [ 1,  3,  4,  4,  4,  5,  7,  7,  9, 13, 13, 15, 20]
    >>> result = list(past_present_future_idx_and_duration_iter(timestamp_seq, 3.5))
    >>> expected = array([[(1, 4, 7), 0.5, 4.5], [(1, 5, 7), 0.5, 5.0], [(1, 5, 8), 1.0, 5.5], [(2, 5, 8), 0.5, 6.5], \
                        [(2, 7, 8), 0.5, 7.0], [(5, 7, 8), 1.0, 7.5], [(6, 7, 8), 0.5, 8.5], \
                        [(6, 8, 8), 0.5, 9.0], [(6, 8, 10), 1.0, 9.5], [(8, 8, 10), 1.0, 10.5], \
                        [(8, 8, 11), 1.0, 11.5], [(9, 8, 11), 0.5, 12.5], [(9, 10, 11), 2.0, 13.0], \
                        [(9, 11, 11), 1.5, 15.0]])
    >>> all(array(list(flatten(result))) == array(list(flatten(expected))))
    True
    >>> window_size = 5
    >>> timestamp_seq = cumsum(random.randint(low=0, high=window_size * 2, size=100))
    >>> result = list(past_present_future_idx_and_duration_iter(timestamp_seq, window_size))
    >>> expected = list(slow_past_present_future_idx_and_duration_iter(timestamp_seq, window_size))
    >>> all(array(list(flatten(result))) == array(list(flatten(expected))))
    True
    """

    if future_range is None:
        future_range = past_range

    # if not timestamps_are_sorted:
    #     timestamp_seq = sorted(timestamp_seq)
    #
    # timestamp_seq = array(timestamp_seq)
    global ppf_idx
    ppf_idx = zeros(3).astype(int)
    global time_to_next
    time_to_next = zeros(3).astype(float)

    idx, dur, timestamp_seq = _idx_and_duration(timestamp_seq, timestamps_are_sorted)
    dur = hstack((dur, 0))
    if DEBUG_LEVEL:
        print(("idx={}\ndur={}\ntimestamp_seq={}".format(idx, dur, timestamp_seq)))
    first_timestamp = timestamp_seq[0]
    # dur = diff(timestamp_seq)
    # idx = arange(len(timestamp_seq)).astype(int)
    n = len(timestamp_seq) - 1

    def _init_state():
        present_timestamp = first_timestamp + past_range

        ppf_idx[_past] = 1
        time_to_next[_past] = dur[0]

        ppf_idx[_present] = where(timestamp_seq <= first_timestamp + past_range)[0][-1]
        time_to_next[_present] = \
            timestamp_seq[ppf_idx[_present] + 1] - first_timestamp - past_range

        ppf_idx[_future] = where(timestamp_seq <= first_timestamp + past_range + future_range)[0][-1]
        time_to_next[_future] = \
            timestamp_seq[ppf_idx[_future] + 1] - first_timestamp - past_range - future_range

        if DEBUG_LEVEL:
            print(("time_to_next={}".format(time_to_next)))
        return present_timestamp, time_to_next.argsort()

    def _shift_dimension(this_idx, shift_by):
        """ Shift a single dimension, updating _idx and _time_to_next"""
        if time_to_next[this_idx] <= shift_by:  # if the next smallest item is the same (<= for float imprecision)
            ppf_idx[this_idx] += 1
            if ppf_idx[this_idx] >= n + 1:
                return None  # should be "caught" by caller: Means "no more further states"
            else:
                if this_idx != 0:
                    time_to_next[this_idx] = dur[ppf_idx[this_idx]]
                    if DEBUG_LEVEL:
                        print(("ppf_idx={}".format(ppf_idx)))
                        print(("time_to_next[{}] = dur[{}] = {}".format(
                            this_idx, ppf_idx[this_idx], dur[ppf_idx[this_idx]])))
                else:
                    time_to_next[this_idx] = dur[ppf_idx[this_idx] - 1]
                    if DEBUG_LEVEL:
                        print(("--ppf_idx={}".format(ppf_idx)))
                        print(("--time_to_next[{}] = dur[{}] = {}".format(
                            this_idx, ppf_idx[this_idx] - 1, dur[ppf_idx[this_idx] - 1])))
        else:
            time_to_next[this_idx] -= shift_by

        return True

    def _shift_state(time_to_next_order, present_timestamp):
        shift_by = time_to_next[time_to_next_order[0]]
        if DEBUG_LEVEL:
            print(('---> shifting by {}'.format(shift_by)))
        present_timestamp += shift_by

        if _shift_dimension(time_to_next_order[0], shift_by) is None:  # shift smallest dimension...
            return None, None  # ... and return None if we're at the end.
        elif _shift_dimension(time_to_next_order[1], shift_by) is None:  # shift next smallest dimension...
            return None, None  # ... and return None if we're at the end.
        elif _shift_dimension(time_to_next_order[2], shift_by) is None:  # shift next smallest dimension...
            return None, None  # ... and return None if we're at the end.

        # next_idx = time_to_next_order[0]
        # time_to_next[next_idx] = dur[ppf_idx[next_idx] - 1]
        # if DEBUG_LEVEL:
        #         print("time_to_next[{}] = dur[{}] = {}".format(
        #             next_idx, ppf_idx[next_idx] - 1, time_to_next[next_idx]))
        return time_to_next.argsort(), present_timestamp  # if you got this far, return the new dimension order

    _present_timestamp, _time_to_next_order = _init_state()

    if DEBUG_LEVEL:
        print("---------------")
        print(("ppf_idx: {}".format(ppf_idx)))
        print(("time_to_next: {}".format(time_to_next)))
        print(("time_to_next_order: {}".format(_time_to_next_order)))
        print(("present_timestamp: {}".format(_present_timestamp)))
    _duration = time_to_next[_time_to_next_order[0]]
    if _duration != 0:
        yield (idx[ppf_idx[_past]][0], idx[ppf_idx[_present]][-1], idx[ppf_idx[_future]][-1]), \
              _duration, \
              _present_timestamp
    while True:
        _time_to_next_order, _present_timestamp = _shift_state(_time_to_next_order, _present_timestamp)
        if _time_to_next_order is None:
            raise StopIteration
        else:
            if DEBUG_LEVEL:
                print("---------------")
                print(("ppf_idx: {}".format(ppf_idx)))
                print(("time_to_next: {}".format(time_to_next)))
                print(("time_to_next_order: {}".format(_time_to_next_order)))
                print(("present_timestamp: {}".format(_present_timestamp)))
            _duration = time_to_next[_time_to_next_order[0]]
            if _duration != 0:
                yield (idx[ppf_idx[_past]][0], idx[ppf_idx[_present]][-1], idx[ppf_idx[_future]][-1]), \
                      _duration, \
                      _present_timestamp


class FeaturePairFactory(object):
    def __init__(self,
                 past_feat_func,
                 past_range,
                 future_feat_func=None,
                 future_range=None,
                 min_timestamp=None,
                 max_timestamp=None,
                 data_is_sorted=False,
                 timestamp_field='timestamp'):

        if future_range is None:
            future_range = past_range
        if future_feat_func is None:
            future_feat_func = past_feat_func

        self.past_feat_func = past_feat_func
        self.future_feat_func = future_feat_func
        self.past_range = past_range
        self.future_range = future_range
        self.min_timestamp = min_timestamp
        self.max_timestamp = max_timestamp
        self.data_is_sorted = data_is_sorted
        if timestamp_field == 'index':
            self.get_timestamps = lambda data: data.index.values
            self.sort_data_according_to_timestamps = lambda data: data.sort_index()
        else:
            self.get_timestamps = lambda data: data[timestamp_field]
            self.sort_data_according_to_timestamps = lambda data: data.sort_values(timestamp_field)

    def _data_in_date_range(self, data):
        lidx = array(self.get_timestamps(data) >= self.min_timestamp) \
               & array(self.get_timestamps(data) < self.max_timestamp)
        return data[lidx]

    def _get_data_iterator(self, data):
        if not self.data_is_sorted:
            data = self.sort_data_according_to_timestamps(data)
        data = self._data_in_date_range(data)

        ppfd = past_present_future_idx_and_duration_iter(
            timestamp_seq=[self.min_timestamp] + self.get_timestamps(data).tolist() + [self.max_timestamp],
            past_range=self.past_range,
            future_range=self.future_range
        )

        for ppf, duration, present_timestamp in ppfd:
            yield data.iloc[ppf[0]:ppf[1]], data.iloc[ppf[1]:ppf[2]], duration, present_timestamp

    def feature_pair_and_duration_iter(self, data):
        data_iterator = self._get_data_iterator(data)
        for past_data, future_data, duration, present_timestamp in data_iterator:
            yield {'past': self.past_feat_func(past_data),
                   'future': self.future_feat_func(future_data),
                   'duration': duration,
                   'present_timestamp': present_timestamp}


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
    return map(_extractor, window_iterator)


def agg_counts(pairs_of_series_iter):
    accum = defaultdict(Counter)

    for pair in pairs_of_series_iter:
        pair_iter = map(lambda x: list(zip(*x)), product(iter(pair[0].to_dict().items()), iter(pair[1].to_dict().items())))
        for x in pair_iter:
            accum[x[0]].update([x[1]])

    return accum