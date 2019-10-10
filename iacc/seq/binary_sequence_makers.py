

__author__ = 'thor'

from numpy import array, zeros


def timedelta_to_integer_hours(t):
    return int(t.total_seconds() / 3600)


def timedelta_to_integer_days(t):
    return int(t.total_seconds() / 86400)


def mk_binary_bins_sequence_from_timestamp_array(ordered_timestamps, time_binning_func=timedelta_to_integer_hours):
    """
    Takes an ordered sequence of timestamps of events, bins them, and returns an array of 0s and 1s where 0 marks
     the occurrence of (at least one) event in the bin.
    Example:
    >>> from datetime import datetime
    >>> from datetime import timedelta
    >>> from numpy import array
    >>>
    >>> delta = datetime(year=1, month=1, day=1, hour=2, minute=45) - datetime(year=1, month=1, day=1, hour=0, minute=0)
    >>> timestamps = array(map(lambda x: x * delta, range(0, 10)))
    >>> print(timestamps)
    [datetime.timedelta(0) datetime.timedelta(0, 9900)
     datetime.timedelta(0, 19800) datetime.timedelta(0, 29700)
     datetime.timedelta(0, 39600) datetime.timedelta(0, 49500)
     datetime.timedelta(0, 59400) datetime.timedelta(0, 69300)
     datetime.timedelta(0, 79200) datetime.timedelta(1, 2700)]
    >>>
    >>> time_binning_func = lambda x: int(x.total_seconds() / 3600)
    >>> mk_binary_bins_sequence_from_timestamp_array(timestamps, time_binning_func)
    array([ 1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
            1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.])
    """
    t = array(list(map(time_binning_func, ordered_timestamps - min(array(ordered_timestamps)))))
    bin_sequence = zeros(t.max() + 1)
    bin_sequence[t] = 1
    return bin_sequence


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print("---> Tests succeeded!")