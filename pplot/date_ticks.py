from __future__ import division

import numpy as np
from datetime import datetime as dt

unit_str_to_unit_in_seconds = {
    'day': 3600 * 24,
    'hour': 3600,
    'h': 3600,
    'minute': 60,
    'mn': 60,
    'second': 1,
    's': 1,
    'millisecond': 0.001,
    'ms': 0.001,
    'microsecond': 1e-6,
    'us': 1e-6,
}

unit_in_seconds = np.array([
    60 * 60 * 24 * 365,  # year
    60 * 60 * 24 * 30,  # month
    60 * 60 * 24 * 7,  # week
    60 * 60 * 24,  # day
    60 * 60,  # hour
    60,  # minute
    1,  # second
    1e-3,  # millisecond
    1e-6,  # microsecond
    1e-9  # nanosecond
])

strftime_format_for_unit = {
    60 * 60 * 24 * 30: '%y-%m-%d',  # month
    60 * 60 * 24 * 7: '%b %d',  # week
    60 * 60 * 24: '%b %d',  # day
    60 * 60: '%d-%H:%M',  # hour
    60: '%H:%M:%S',  # minute
    1: '%M:%S.%f',  # second
    1e-3: "%S''%f",  # millisecond
    1e-6: "%S''%f",  # microsecond
}


def utc_datetime_from_val_and_unit(val, unit):
    if isinstance(unit, basestring):
        unit = unit_str_to_unit_in_seconds[unit]
    return dt.utcfromtimestamp(val * unit)


def largest_unit_that_changes_at_every_tick(ticks, ticks_unit):
    """
    Returns the largest time unit for which each time tick changes.
    :param ticks: The list of ticks
    :param ticks_unit: The unit of the elements of ticks, expressed in seconds. For example, if the list
        contains hours, unit=3600, if minutes, unit=60, if seconds unit=1,
        if milliseconds unit=0.001.
            Note: You can also use a string to express the unit, as long as it's recognized by the
            unit_str_to_unit_in_seconds dict. Keys recognized:
            ['day', 'hour', 'h', 'minute', 'mn', 'second', 's', 'millisecond', 'ms', 'microsecond', 'us']
    :return:
    """
    ticks = np.array(ticks)
    if isinstance(ticks_unit, basestring):
        ticks_unit = unit_str_to_unit_in_seconds[ticks_unit]
    min_tick_diff = min(np.diff(ticks))
    min_tick_diff *= ticks_unit  # convert to seconds

    for u in unit_in_seconds:
        if u < min_tick_diff:
            return u


def strftime_format_for_ticks(ticks, ticks_unit):
    unit = largest_unit_that_changes_at_every_tick(ticks, ticks_unit)
    return strftime_format_for_unit[unit]


def str_ticks(ticks, ticks_unit):
    t_format = strftime_format_for_ticks(ticks, ticks_unit)
    return map(lambda x: utc_datetime_from_val_and_unit(x, ticks_unit).strftime(t_format), ticks)


def unit_aligned_ticks(ticks, ticks_unit):
    pass
