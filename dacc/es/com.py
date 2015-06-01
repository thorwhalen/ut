__author__ = 'thor'

import sys
import elasticsearch
from elasticsearch import Elasticsearch
import elasticsearch.helpers

from pymongo import MongoClient
import pandas as pd


class ElasticCom(object):

    def __init__(self, index, doc_type, hosts='localhost:9200', **kwargs):
        self.index = index
        self.doc_type = doc_type
        self.es = Elasticsearch(hosts=hosts, **kwargs)

    def search_and_export_to_dict(self, *args, **kwargs):
        _id = kwargs.pop('_id', True)
        data_key = kwargs.pop('data_key', kwargs.get('fields')) or '_source'
        kwargs = dict({'index': self.index, 'doc_type': self.doc_type}, **kwargs)
        print args
        print kwargs
        if kwargs.get('size', None) is None:
            kwargs['size'] = 1
            t = self.es.search(*args, **kwargs)
            kwargs['size'] = t['hits']['total']
        return get_search_hits(self.es.search(*args, **kwargs), _id=_id, data_key=data_key)

    def search_and_export_to_df(self, *args, **kwargs):
        convert_numeric = kwargs.pop('convert_numeric', True)
        convert_dates = kwargs.pop('convert_dates', 'coerce')
        exclude_fields = kwargs.pop('exclude_fields', [])

        mapping = self.es.indices.get_mapping(index=self.index, doc_type=self.doc_type)
        mapping = mapping[self.index]['mappings'][self.doc_type]['properties']
        fields_with_type = es_types_to_main_types(mapping)

        df = pd.DataFrame(self.search_and_export_to_dict(*args, **kwargs))
        if len(exclude_fields) > 0:
            if isinstance(exclude_fields, basestring):
                exclude_fields = [exclude_fields]
            for field in exclude_fields:
                df.drop(field, axis=1, inplace=True)

        if convert_numeric:
            fields = [x for x in fields_with_type['number'] if x not in exclude_fields]
            if len(fields) > 0:
                df[fields] = df[fields].convert_objects(convert_numeric=convert_numeric, copy=False)
        if convert_dates:
            fields = [x for x in fields_with_type['date'] if x not in exclude_fields]
            if len(fields) > 0:
                df[fields] = df[fields].convert_objects(convert_dates=convert_dates, copy=False)
        return df

    def insert(self, d, overwrite=False, **kwargs):
        doc_id = str(d.pop('_id'))
        self.es.create(index=self.index, doc_type=self.doc_type, body=d, id=doc_id, **kwargs)

    def import_from_mongo_cursor(self, cursor):
        def action_gen():
            for doc in cursor:
                op_dict = {
                    '_index': self.index,
                    '_type': self.doc_type,
                    '_id': to_utf8_or_bust(doc['_id'])
                }
                doc.pop('_id')
                op_dict['_source'] = doc
                yield op_dict

        res = elasticsearch.helpers.bulk(self.es, action_gen())
        return res

    def import_mongo_collection(self, mongo_db, mongo_collection, **kwargs):
        return self.import_from_mongo_cursor(MongoClient()[mongo_db][mongo_collection].find(**kwargs))


def es_types_to_main_types(mapping):
    fields_with_type = dict()
    fields_with_type['number'] = [k for k, v in mapping.iteritems() if v['type']
                                   in ['float', 'double', 'byte', 'short', 'integer', 'long']]
    fields_with_type['string'] = [k for k, v in mapping.iteritems() if v['type'] in ['string']]
    fields_with_type['date'] = [k for k, v in mapping.iteritems() if v['type'] in ['date']]
    return fields_with_type


def get_search_hits(es_response, _id=True, data_key=None):
    response_hits = es_response['hits']['hits']
    if len(response_hits) > 0:
        if data_key is None:
            for hit in response_hits:
                if '_source' in hit.keys():
                    data_key = '_source'
                    break
                elif 'fields' in hit.keys():
                    data_key = 'fields'
                    break
            if data_key is None:
                raise ValueError("Neither _source nor fields were in response hits")

        if _id is False:
            return [x.get(data_key, None) for x in response_hits]
        else:
            return [dict(_id=x['_id'], **x.get(data_key, {})) for x in response_hits]
    else:
        return []


def to_utf8_or_bust(obj):
    if isinstance(obj, basestring):
        try:
            return obj.encode('utf-8')
        except UnicodeDecodeError:
            return obj.decode('ISO-8859-1').encode('utf-8')
    else:
        try:
            return to_utf8_or_bust(str(obj))
        except (UnicodeDecodeError, UnicodeEncodeError, TypeError):
            return to_utf8_or_bust(unicode(obj))


def stringify_when_necessary(d, fields_to_stringify):
    d.update({k: str(d[k]) for k in fields_to_stringify})
    return d

# closure for displaying status of operation
def show_status(current_count, total_count):
    percent_complete = current_count * 100 / total_count
    sys.stdout.write("\rstatus: %d%%" % percent_complete)
    sys.stdout.flush()



# NOTES:

# Deleting an index:
#   es_client.indices.delete(index = index_name, ignore=[400, 404])

# Creating an index:
#   es_client.indices.create(index = index_name)