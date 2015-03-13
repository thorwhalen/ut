__author__ = 'thorwhalen'

from datetime import timedelta
from datetime import datetime


def daterange(start_date, end_date):
    for n in range((end_date - start_date).days):
        yield start_date + timedelta(n)


def datetimes_ranges_defining_months(from_date, to_date):
    from_year = from_date.year
    from_month = from_date.month
    to_year = to_date.year
    to_month = to_date.month


