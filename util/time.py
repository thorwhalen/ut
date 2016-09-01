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


def utcnow_timestamp():
    return (dt.utcnow() - dt.utcfromtimestamp(0)).total_seconds()