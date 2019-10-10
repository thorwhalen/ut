

__author__ = 'thor'

from collections import defaultdict
from numpy import array


def simple_dict_event_extractor(row, condition_for_creating_event, id_field, timestamp_field, name_of_event):
    """
    Takes a row of the data df and returns an event record {id, event, timestamp}
    if the row satisfies the condition (i.e. condition_for_creating_event(row) returns True)
    """
    if condition_for_creating_event(row):
        return {'id': row[id_field], 'event': name_of_event, 'timestamp': row[timestamp_field]}


def group_event_info_by_id(df, event_extractor, id_field='id'):
    """
    Takes a data df, extract event dicts and construct id grouped event info sequences
    Input:
        * df: the data as a dataframe whose rows represent events
        * event_extractor: the function that takes
    """
    seq = defaultdict(list)  # key should be an id and value a list of dates
    for _, row in df.iterrows():
        event_item = event_extractor(row)
        if event_item:
            # pop the id_field into a key of seq and append the remaining to the list
            seq[event_item.pop(id_field)].append(event_item)

    return {k: array(sorted(v)) for k, v in seq.items()}


def mk_id_keyed_dict_of_event_timestamps(df, event_extractor, id_field='id', timestamp_field='timestamp'):
    """
    Takes a data df and returns a dict whose keys are id_fields and whose values are ordered lists of timestamps
    of events.
    Input:
        * df: the data as a dataframe whose rows represent events
        * event_extractor: the function that takes
    """
    seq = defaultdict(list)  # key should be an id and value a list of dates
    for _, row in df.iterrows():
        event_item = event_extractor(row)
        if event_item:
            seq[event_item[id_field]].append(event_item[timestamp_field])

    return {k: array(sorted(v)) for k, v in seq.items()}





