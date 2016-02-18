from __future__ import division

__author__ = 'thor'

import json


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