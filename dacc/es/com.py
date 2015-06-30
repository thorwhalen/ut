__author__ = 'thor'

import sys
import elasticsearch
from elasticsearch import Elasticsearch
import elasticsearch.helpers
from elasticsearch.helpers import scan

from collections import defaultdict

from pymongo import MongoClient
import pandas as pd
from itertools import imap, islice

from ut.pdict.manip import rollout


class ElasticCom(object):

    def __init__(self, index, doc_type, hosts='localhost:9200', **kwargs):
        self.index = index
        self.doc_type = doc_type
        self.es = Elasticsearch(hosts=hosts, **kwargs)

    def search(self, *args, **kwargs):
        kwargs = dict({'index': self.index, 'doc_type': self.doc_type}, **kwargs)
        return self.es.search(*args, **kwargs)

    def source_scan_iterator(self, extractor=None, scroll='10m', *args, **kwargs):
        """
        Returns an iterator that yields the _source field of scan results one at a time
        """
        kwargs['index'] = kwargs.get('index', self.index)
        kwargs['doc_type'] = kwargs.get('doc_type', self.doc_type)
        start = kwargs.pop('start', None)
        stop = kwargs.pop('stop', None)
        scanner = scan(self.es, scroll=scroll, *args, **kwargs)
        if stop is not None:
            scanner = islice(scan(self.es, scroll=scroll, *args, **kwargs), start, stop)

        if extractor is None:
            return imap(lambda x: x['_source'], scanner)
        else:
            return imap(lambda x: extractor(x['_source']), scanner)

    def search_and_export_to_dict(self, *args, **kwargs):
        _id = kwargs.pop('_id', True)
        data_key = kwargs.pop('data_key', kwargs.get('fields')) or '_source'
        rollout_key = kwargs.pop('rollout_key', None)

        kwargs = dict({'index': self.index, 'doc_type': self.doc_type}, **kwargs)

        # If size is None, set it to hits total
        if kwargs.get('size', None) is None:
            kwargs['size'] = 1
            t = self.es.search(*args, **kwargs)
            kwargs['size'] = t['hits']['total']

        d = get_search_hits(self.es.search(*args, **kwargs), _id=_id, data_key=data_key)

        if rollout_key:
            d = rollout(d, key=rollout_key, copy=False)

        return d

    def search_and_export_to_df(self, *args, **kwargs):
        convert_numeric = kwargs.pop('convert_numeric', True)
        convert_dates = kwargs.pop('convert_dates', 'coerce')  # specify convert_dates='coerce' to convert dates

        exclude_fields = kwargs.pop('exclude_fields', [])

        df = pd.DataFrame(self.search_and_export_to_dict(*args, **kwargs))
        if len(exclude_fields) > 0:
            if isinstance(exclude_fields, basestring):
                exclude_fields = [exclude_fields]
            for field in exclude_fields:
                df.drop(field, axis=1, inplace=True)

        if convert_numeric or convert_dates:
            mapping = self.get_fields_mapping_info()
            fields_with_type = es_types_to_main_types(mapping)

            if convert_numeric:
                fields = [x for x in fields_with_type['number'] if x not in exclude_fields]
                if len(fields) > 0:
                    df[fields] = df[fields].convert_objects(convert_numeric=convert_numeric, copy=False)

            if convert_dates:
                fields = [x for x in fields_with_type['date'] if x not in exclude_fields]
                if len(fields) > 0:
                    df[fields] = df[fields].convert_objects(convert_dates=convert_dates, copy=False)

        return df

    def get_fields_mapping_info(self):
        mapping = self.es.indices.get_mapping(index=self.index, doc_type=self.doc_type)
        mapping = mapping[self.index]['mappings'][self.doc_type]['properties']
        return mapping

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
    fields_with_type = defaultdict(list)
    for k, v in mapping.iteritems():
        if 'type' in v.keys():
            v_type = v.get('type')
            if v_type in ['float', 'double', 'byte', 'short', 'integer', 'long']:
                fields_with_type['number'].append(k)
            else:
                fields_with_type[v_type].append(k)
        elif 'properties' in v.keys():
            nested_fields_with_type = es_types_to_main_types(v['properties'])
            for kk, vv in nested_fields_with_type.iteritems():
                fields_with_type[kk].extend(map(lambda x: k + '.' + x, vv))
    # fields_with_type['number'] = [k for k, v in mapping.iteritems() if v['type']
    #                               in ['float', 'double', 'byte', 'short', 'integer', 'long']]
    # fields_with_type['string'] = [k for k, v in mapping.iteritems() if v['type'] in ['string']]
    # fields_with_type['date'] = [k for k, v in mapping.iteritems() if v['type'] in ['date']]
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