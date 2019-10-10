__author__ = 'thorwhalen'

from datetime import timedelta
from datetime import datetime
from pytz import timezone


def function_to_convert_to_timezone_from_timezone(to_timezone, from_timezone='UTC'):
    if isinstance(to_timezone, str):
        to_timezone = timezone(to_timezone)
    if isinstance(from_timezone, str):
        from_timezone = timezone(from_timezone)

    return lambda x: from_timezone.localize(x).astimezone(to_timezone)


def mk_time_info_extractor(spec):
    """
    Returns a function that will extract information from timestamps in a dict format.
    The specification should be a list of timetuple attributes
    (see https://docs.python.org/2/library/time.html#time.struct_time) to extract,
    or a {k: v, ...} dict where v are the timetuple attributes, and k what you want to call them in the output.

    Example:
        fun = mk_time_info_extractor({'day_of_week': 'tm_wday', 'hour_of_day': 'tm_hour'})
        # assuming you defined some timestamp called t...
        print t
        print fun(t)
    2015-06-02 20:46:16.629000
    {'day_of_week': 1, 'hour_of_day': 20}
    """
    if not isinstance(spec, dict):  # if spec is not a dict, make it so
        spec = {x: x for x in spec}

    def extractor(timestamp):
        time_struct = timestamp.timetuple()
        return {k: time_struct.__getattribute__(v) for k, v in spec.items()}

    return extractor


day_of_week_strings = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']


def day_of_week_integer_to_string(day_of_week_integer):
    return day_of_week_strings[day_of_week_integer]


def daterange(start_date, end_date):
    for n in range((end_date - start_date).days):
        yield start_date + timedelta(n)


def datetimes_ranges_defining_months(from_date, to_date):
    from_year = from_date.year
    from_month = from_date.month
    to_year = to_date.year
    to_month = to_date.month

