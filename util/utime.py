import datetime  # keep around! It's to be able to access the mother of datetime
from datetime import datetime as dt
from dateutil import tz

__author__ = 'thor'

second_ms = 1000.0
second_ns = 1e9
minute_ms = float(60 * second_ms)
five_mn_ms = 5 * minute_ms
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


def utcnow_ns():
    return (dt.utcnow() - epoch).total_seconds() * second_ns


def utc_datetime_to_utc_ms(utc_datetime):
    return (utc_datetime - epoch).total_seconds() * second_ms


def utc_ms_to_utc_datetime(ums):
    return dt.utcfromtimestamp(ums / second_ms)


def utc_ms_to_local_datetime(ums):
    return dt.fromtimestamp(ums / second_ms)


def utc_to_local(utc_date):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    return utc_date.replace(tzinfo=from_zone).astimezone(to_zone)


def local_to_utc(local_date):
    from_zone = tz.tzlocal()
    to_zone = tz.tzutc()
    return local_date.replace(tzinfo=from_zone).astimezone(to_zone)


def day_utc_ms_from_utc_ms(ums):
    """
    Get a utc_ms corresponding to midnight of the day of the input ums
    :param ums: utc in milliseconds
    :return: utc_ms corresponding to midnight of the day of the input ums
    >>> from numpy.random import randint
    >>> ums = randint(1, 2000000000000)
    >>> day_ums = utc_datetime_to_utc_ms(day_datetime_from_utc_ms(ums))
    >>> int(day_utc_ms_from_utc_ms(day_ums)) == int(day_ums)
    True
    """
    return int(day_ms * (ums // day_ms))


def day_datetime_from_datetime(date_time):
    """
    Get a datetime corresponding to midnight of the day of the input datetime
    :param date_time: (utc) datetime
    :return:
    >>> from numpy.random import randint
    >>> ums = randint(1, 2000000000000)
    >>> date_time = utc_ms_to_utc_datetime(ums)
    >>> day_ums = day_utc_ms_from_utc_ms(ums)
    >>> day_datetime_from_datetime(date_time) == utc_ms_to_utc_datetime(day_ums)
    True
    """
    return datetime.datetime(date_time.year, date_time.month, date_time.day)


def day_datetime_from_utc_ms(ums):
    """
    Get a datetime corresponding to midnight of the day of the input ums
    :param ums: utc in milliseconds
    :return: datetime corresponding to midnight of the day of the input ums
    >>> from numpy.random import randint
    >>> ums = randint(1, 2000000000000)
    >>> day_datetime = utc_ms_to_utc_datetime(day_utc_ms_from_utc_ms(ums))
    >>> day_datetime_from_utc_ms(ums) == day_datetime
    True
    """
    dt = utc_ms_to_utc_datetime(ums)
    return datetime.datetime(dt.year, dt.month, dt.day)


#################### Display

def seconds_to_mmss_str(s):
    return "{:.0f}m{:02.0f}s".format(s / 60, s % 60)


#################### Deprecated

def unix_time_ms_to_datetime(ums):
    raise DeprecationWarning("Use utc_ms_to_local_datetime instead")


def datetime_to_unix_time_ms(date):
    raise DeprecationWarning("Use utc_datetime_to_utc_ms instead")
