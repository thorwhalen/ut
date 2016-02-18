from __future__ import division

__author__ = 'thor'

import ut.dacc.es.queries as es_queries


def min_and_max_of_field(escom, field='timestamp'):
    result = escom.search(body=es_queries.min_and_max_of_field(field=field))
    return result['aggregations']['min_of_field']['value'], result['aggregations']['max_of_field']['value']


# def random_search(escom, timestamp_field='timestamp'):
#     pass