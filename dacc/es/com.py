__author__ = 'thor'

import sys
import elasticsearch
from elasticsearch import Elasticsearch
import elasticsearch.helpers

from pymongo import MongoClient


class ElasticCom(object):

    def __init__(self, index, doc_type, hosts='localhost:9200', **kwargs):
        self.index = index
        self.doc_type = doc_type
        self.es = Elasticsearch(hosts=hosts, **kwargs)

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