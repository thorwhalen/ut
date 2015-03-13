__author__ = 'thor'

import datetime


def find_latest_docs(collection, num_of_docs, output='cursor'):
    pass


def find_new_docs(collection=None, date_key=None, thresh_date=None, output='cursor'):
    assert date_key
    assert thresh_date
    if isinstance(thresh_date, int):  # if thresh_date is an int, consider it to be the number of days than defines "new"
        greater_or_equal = datetime.datetime.now() - datetime.timedelta(thresh_date)
    query = {date_key: {'$gte': greater_or_equal}}
    if output == 'query':
        return query
    else:
        return collection.find(query)


def find_old_docs(collection=None, date_key=None, thresh_date=None, output='cursor'):
    assert date_key
    assert thresh_date
    if isinstance(thresh_date, int):  # if thresh_date is an int, consider it to be the number of days than defines "old"
        thresh_date = datetime.datetime.now() - datetime.timedelta(thresh_date)
    query = {date_key: {'$lt': thresh_date}}
    if output == 'query':
        return query
    else:
        return collection.find(query)



