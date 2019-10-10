

__author__ = 'thor'

import json


def more_like(_index, _type, item_list, similarity_fields, item_field='_id', **more_like_this_kwargs):
    if isinstance(similarity_fields, str):
        similarity_fields = [similarity_fields]
    like_these = [{'_index': _index, '_type': _type, item_field: x} for x in item_list]

    more_like_this_query = {
        "fields": similarity_fields,
        "like": like_these
    }
    more_like_this_query.update(**more_like_this_kwargs)

    return {
        "query": {
            "more_like_this": more_like_this_query
        }
    }


def match_field(field, value=None):
    if value is None:
        assert isinstance(field, dict), "you must specify a field and value, or field={field: value}"
        field, value = next(iter(field.items()))
    return {
        "query": {
            "match": {
                field: value
            }
        }
    }


def min_and_max_of_field(field='timestamp'):
    return \
        {
            "size": 0,
            "query": {
                "match_all": {}
            },
            "aggs": {
                "min_of_field": {
                    "min": {
                        "field": field
                    }
                },
                "max_of_field": {
                    "max": {
                        "field": field
                    }
                }
            }
        }


def field_range(field='timestamp', gte=0, lte=1e6):
    return {
        "query": {
            "range": {
                field: {
                    "gte": gte,
                    "lte": lte
                }
            }
        }
    }


def field_gte(field='timestamp', gte=0):
    return {
        "query": {
            "range": {
                field: {
                    "gte": gte
                }
            }
        }
    }


def fields_exist(fields):
    return {"filter": {"exists": {"field": fields}}}
