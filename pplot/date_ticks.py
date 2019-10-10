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
    if isinstance(unit, str):
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
    if isinstance(ticks_unit, str):
        ticks_unit = unit_str_to_unit_in_seconds[ticks_unit]
    min_tick_diff = min(np.diff(ticks))
    min_tick_diff *= ticks_unit  # convert to seconds

    for u in unit_in_seconds:
        if u < min_tick_diff:
            return u


def strftime_format_for_ticks(ticks, ticks_unit):
    unit = largest_unit_that_changes_at_every_tick(ticks, ticks_unit)
    return strftime_format_for_unit[unit]


def strftime_with_precision(tick, format, sub_secs_precision=2):
    """
    Returns a formatted string for a datetime (tick).
    :param tick: The datetime for this tick
    :param format: The formatting string
    :param sub_secs_precision: Number of digits to used for sub-seconds.
        If None, will choose it "smartly/dynamically"
    :return: Formatted string
    """
    t = tick.strftime(format)
    is_us = '%f' in format
    if is_us:
        if sub_secs_precision is None:
            while t[-1] == '0':
                t = t[:-1]
            while not t[-1].isdigit():
                t = t[:-1]
            return t
        else:
            if sub_secs_precision < 0:
                sub_secs_precision = 0
            elif sub_secs_precision > 6:
                sub_secs_precision = 6

            DFLT_PRECISION = 6
            digits_to_skip = DFLT_PRECISION - sub_secs_precision
            if digits_to_skip == 0:
                return t
            else:
                t = t[:-digits_to_skip]
                while not t[-1].isdigit():
                    t = t[:-1]
                return t
    else:
        return t


def str_ticks(ticks, ticks_unit, sub_secs_precision=2):
    t_format = strftime_format_for_ticks(ticks, ticks_unit)
    return [strftime_with_precision(utc_datetime_from_val_and_unit(x, ticks_unit), t_format, sub_secs_precision) for x
            in ticks]


def use_time_ticks(ax=None, ticks_unit=0.001):
    from matplotlib.pylab import gca
    if ax is None:
        ax = gca()
    _xticks = ax.get_xticks()
    ax.set_xticks(_xticks)
    ax.set_xticklabels(str_ticks(ticks=_xticks, ticks_unit=ticks_unit))
    ax.margins(x=0)


def unit_aligned_ticks(ticks, ticks_unit):
    pass
