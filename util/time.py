from __future__ import division

from datetime import datetime as dt

__author__ = 'thor'

second_ms = 1000.0
minute_ms = float(60 * 1e9)
hour_ms = float(60 * minute_ms)
day_ms = float(24 * hour_ms)
day_hours = float(24)

hour_as_day = 1 / day_hours
day_minutes = float(24 * 60)
minute_as_day = 1 / day_minutes
day_seconds = float(24 * 3600)
second_as_day = 1 / day_seconds

epoch = dt.utcfromtimestamp(0)


def utcnow_timestamp():
    return (dt.utcnow() - epoch).total_seconds()


def utcnow_ms():
    return (dt.utcnow() - epoch).total_seconds() * second_ms


def utc_datetime_to_utc_ms(utc_datetime):
    return (utc_datetime - epoch).total_seconds() * second_ms


def utc_ms_to_utc_datetime(ts):
    return dt.utcfromtimestamp(ts / second_ms)


def utc_ms_to_local_datetime(ts):
    return dt.fromtimestamp(ts / second_ms)


#################### Display

def seconds_to_mmss_str(s):
    return "{:.0f}m{:02.0f}s".format(s / 60, s % 60)


#################### Deprecated

def unix_time_ms_to_datetime(ts):
    raise DeprecationWarning("Use utc_ms_to_local_datetime instead")


def datetime_to_unix_time_ms(date):
    raise DeprecationWarning("Use utc_datetime_to_utc_ms instead")
