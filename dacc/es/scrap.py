__author__ = 'thor'

from elasticsearch import Elasticsearch
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
        if kwargs.get('size', None) is None:
            kwargs['size'] = 1
            t = self.es.search(*args, **kwargs)
            kwargs['size'] = t['hits']['total']

        return get_search_hits(self.es.search(*args, **kwargs), _id=_id, data_key=data_key)

    def search_and_export_to_df(self, *args, **kwargs):
        convert_numeric = kwargs.pop('convert_numeric', True)
        convert_dates = kwargs.pop('convert_dates', 'coerce')
        df = pd.DataFrame(self.search_and_export_to_dict(*args, **kwargs))
        if convert_numeric:
            df = df.convert_objects(convert_numeric=convert_numeric, copy=True)
        if convert_dates:
            df = df.convert_objects(convert_dates=convert_dates, copy=True)
        return df


def get_search_hits(es_response, _id=True, data_key=None):
    response_hits = es_response['hits']['hits']
    if len(response_hits) > 0:
        if data_key is None:
            for hit in response_hits:
                if '_source' in list(hit.keys()):
                    data_key = '_source'
                    break
                elif 'fields' in list(hit.keys()):
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



# NOTES:

# Deleting an index:
#   es_client.indices.delete(index = index_name, ignore=[400, 404])

# Creating an index:
#   es_client.indices.create(index = index_name)
