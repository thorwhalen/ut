"""Elastic Search access"""
__author__ = 'thor'

import sys
import numpy as np
import elasticsearch
from elasticsearch import Elasticsearch
import elasticsearch.helpers
from elasticsearch.helpers import scan

from collections import defaultdict

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import pandas as pd
from itertools import islice, chain

from ut.pdict.manip import rollout
from ut.util.log import printProgress
from ut.util.uiter import print_iter_progress
import ut.dacc.es.queries as es_queries
import ut.dacc.es.get as es_get


class ElasticCom(object):
    def __init__(self, index, doc_type=None, hosts='localhost:9200', **kwargs):
        self.index = index
        self.doc_type = doc_type
        self.es = Elasticsearch(hosts=hosts, **kwargs)

    @property
    def _index_and_doc_type(self):
        return {'index': self.index, 'doc_type': self.doc_type}

    def count(self, *args, **kwargs):
        kwargs = dict({'index': self.index, 'doc_type': self.doc_type}, **kwargs)
        return self.es.count(*args, **kwargs)['count']

    def search(self, *args, **kwargs):
        kwargs = dict({'index': self.index, 'doc_type': self.doc_type}, **kwargs)
        return self.es.search(*args, **kwargs)

    def sample_with_arithmetic_seq(self, sample_size, initial_idx=None, **kwargs):
        """
        Selects docs indexed by an arithmetic sequence (initial_idx + i * step) where step is chosen in order to make
        the coverage of the arithmetic sequence maximal.
        """
        ndocs = self.count()
        step = int((ndocs - initial_idx) / sample_size)
        initial_idx = initial_idx or int(
            (ndocs - sample_size * step) / 2
        )  # choose initial_idx (if not given) so it ~=end
        end = initial_idx + sample_size * step
        return islice(self.scan_extractor(**kwargs), initial_idx, end, step)

    def sample_based_on_random_field_val(
        self,
        n_rand_picks=10,
        rand_field='timestamp',
        rand_batch_size=1,
        *args,
        **kwargs
    ):
        UserWarning(
            'sample_based_on_random_field_val has not been verified and may have bugs'
        )
        min_random_field_val, max_random_field_val = es_get.min_and_max_of_field(
            self, field=rand_field
        )
        range = max_random_field_val - min_random_field_val

        body = kwargs.get('body', {})

        def mk_rand_body():
            gte = min_random_field_val + np.random.rand() * range
            gte_body = es_queries.field_gte(field=rand_field, gte=gte)
            body['query'] = gte_body['query']
            # print(body)
            return body

        return chain(
            *map(
                lambda x: self.search_and_export_to_dict(
                    body=mk_rand_body(), size=rand_batch_size, *args, **kwargs
                ),
                range(n_rand_picks),
            )
        )

    def scan_extractor(
        self,
        query=None,
        extractor=None,
        get_item='_source',
        print_progress_every=None,
        *args,
        **kwargs
    ):
        """
        Returns an iterator that yields a function (the extractor) of the get_item field of scan results one at a time.
        """
        kwargs['index'] = kwargs.get('index', self.index)
        kwargs['doc_type'] = kwargs.get('doc_type', self.doc_type)
        start = kwargs.pop('start', None)
        stop = kwargs.pop('stop', None)
        scroll = kwargs.pop('scroll', '10m')

        # if stop is not None and 'size' in kwargs:
        #     if start is None:
        #         start = 0
        #     stop = kwargs.get('size') + start

        if start is not None or stop is not None:
            start = start or 0
            scanner = islice(
                scan(self.es, query=query, scroll=scroll, *args, **kwargs), start, stop
            )
        else:
            scanner = scan(self.es, query=query, scroll=scroll, *args, **kwargs)

        if extractor is None:
            if get_item is None:
                extractor_ = lambda x: x
            else:
                extractor_ = lambda x: x[get_item]
        else:
            if get_item is None:
                extractor_ = lambda x: extractor(x)
            else:
                extractor_ = lambda x: extractor(x[get_item])

        if print_progress_every is None:
            return map(extractor_, scanner)
        else:
            return print_iter_progress(
                map(extractor_, scanner), print_progress_every=print_progress_every
            )

    def source_scan_iterator(
        self, extractor=None, print_progress_every=None, *args, **kwargs
    ):
        """
        Returns an iterator that yields a function (the extractor) of the _source field of scan results one at a time.
        """
        kwargs['index'] = kwargs.get('index', self.index)
        kwargs['doc_type'] = kwargs.get('doc_type', self.doc_type)
        start = kwargs.pop('start', None)
        stop = kwargs.pop('stop', None)
        scroll = kwargs.pop('scroll', '10m')
        scanner = scan(self.es, scroll=scroll, *args, **kwargs)
        if stop is not None:
            scanner = islice(scan(self.es, scroll=scroll, *args, **kwargs), start, stop)

        if extractor is None:
            if print_progress_every is None:
                return map(lambda x: x['_source'], scanner)
            else:
                return print_iter_progress(
                    map(lambda x: x['_source'], scanner),
                    print_progress_every=print_progress_every,
                )
        else:
            if print_progress_every is None:
                return map(lambda x: extractor(x['_source']), scanner)
            else:
                return print_iter_progress(
                    map(lambda x: extractor(x['_source']), scanner),
                    print_progress_every=print_progress_every,
                )

    def dict_of_source_data(
        self,
        extractor=None,
        start=None,
        stop=None,
        print_progress_every=None,
        source_scan_iterator_kwargs={},
    ):

        source_scan_iterator_kwargs['scroll'] = source_scan_iterator_kwargs.get(
            'scroll', '10m'
        )
        extracting_scanner = self.source_scan_iterator(
            extractor=extractor, start=start, stop=stop, **source_scan_iterator_kwargs
        )
        print_progress_every = print_progress_every or np.inf
        start = start or 0

        d = list()
        for i, item in enumerate(extracting_scanner, start=start):
            if np.mod(i, print_progress_every) == 0:
                printProgress('offset: {}'.format(i))
            if item is not None:
                d.append(item)

        return d

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
        convert_dates = kwargs.pop(
            'convert_dates', 'coerce'
        )  # specify convert_dates='coerce' to convert dates

        exclude_fields = kwargs.pop('exclude_fields', [])

        df = pd.DataFrame(self.search_and_export_to_dict(*args, **kwargs))
        if len(exclude_fields) > 0:
            if isinstance(exclude_fields, str):
                exclude_fields = [exclude_fields]
            for field in exclude_fields:
                df.drop(field, axis=1, inplace=True)

        if convert_numeric or convert_dates:
            mapping = self.get_fields_mapping_info()
            fields_with_type = es_types_to_main_types(mapping)

            if convert_numeric:
                fields = [
                    x for x in fields_with_type['number'] if x not in exclude_fields
                ]
                if len(fields) > 0:
                    df[fields] = df[fields].convert_objects(
                        convert_numeric=convert_numeric, copy=False
                    )

            if convert_dates:
                fields = [
                    x for x in fields_with_type['date'] if x not in exclude_fields
                ]
                if len(fields) > 0:
                    df[fields] = df[fields].convert_objects(
                        convert_dates=convert_dates, copy=False
                    )

        return df

    def get_doc_types(self):
        mapping = self.es.indices.get_mapping(index=self.index)
        return list(mapping[self.index]['mappings'].keys())

    def get_fields_mapping_info(self):
        mapping = self.es.indices.get_mapping(index=self.index, doc_type=self.doc_type)
        if self.doc_type is None:
            return mapping[self.index]['mappings']
        else:
            return mapping[self.index]['mappings'][self.doc_type]['properties']

    def insert(self, d, overwrite=False, **kwargs):
        doc_id = str(d.pop('_id'))
        self.es.create(
            index=self.index, doc_type=self.doc_type, body=d, id=doc_id, **kwargs
        )

    def import_from_mongo_cursor(self, cursor, doc_func=None):
        def action_gen():
            for doc in cursor:
                op_dict = {
                    '_index': self.index,
                    '_type': self.doc_type,
                    '_id': to_utf8_or_bust(doc['_id']),
                }
                doc.pop('_id')
                if doc_func is None:
                    op_dict['_source'] = doc
                else:
                    op_dict['_source'] = doc_func(doc)
                yield op_dict

        res = elasticsearch.helpers.bulk(self.es, action_gen())
        return res

    def import_mongo_collection(
        self, mongo_db=None, mongo_collection=None, doc_func=None, **kwargs
    ):
        if mongo_db is not None:
            if isinstance(mongo_db, str):
                mongo_db = MongoClient()[mongo_db]
        else:  # mongo_db is None
            if mongo_collection is None:
                raise ValueError(
                    "You can't have both mongo_db and mongo_collection be None. I need SOMETHING to go by!"
                )

        if isinstance(mongo_db, Collection):
            mgc = mongo_db
        elif isinstance(mongo_collection, Collection):
            mgc = mongo_collection
        else:
            mgc = mongo_db[mongo_collection]

        return self.import_from_mongo_cursor(mgc.find(**kwargs), doc_func=doc_func)

    def head(self, n_entres=5):
        return self.search_and_export_to_dict(size=n_entres)


def es_types_to_main_types(mapping):
    fields_with_type = defaultdict(list)
    for k, v in mapping.items():
        if 'type' in list(v.keys()):
            v_type = v.get('type')
            if v_type in ['float', 'double', 'byte', 'short', 'integer', 'long']:
                fields_with_type['number'].append(k)
            else:
                fields_with_type[v_type].append(k)
        elif 'properties' in list(v.keys()):
            nested_fields_with_type = es_types_to_main_types(v['properties'])
            for kk, vv in nested_fields_with_type.items():
                fields_with_type[kk].extend([k + '.' + x for x in vv])
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
                if '_source' in list(hit.keys()):
                    data_key = '_source'
                    break
                elif 'fields' in list(hit.keys()):
                    data_key = 'fields'
                    break
            if data_key is None:
                raise ValueError('Neither _source nor fields were in response hits')

        if _id is False:
            return [x.get(data_key, None) for x in response_hits]
        else:
            return [dict(_id=x['_id'], **x.get(data_key, {})) for x in response_hits]
    else:
        return []


def to_utf8_or_bust(obj):
    if isinstance(obj, str):
        try:
            return obj.encode('utf-8')
        except UnicodeDecodeError:
            return obj.decode('ISO-8859-1').encode('utf-8')
    else:
        try:
            return to_utf8_or_bust(str(obj))
        except (UnicodeDecodeError, UnicodeEncodeError, TypeError):
            return to_utf8_or_bust(str(obj))


def stringify_when_necessary(d, fields_to_stringify):
    d.update({k: str(d[k]) for k in fields_to_stringify})
    return d


# closure for displaying status of operation
def show_status(current_count, total_count):
    percent_complete = current_count * 100 / total_count
    sys.stdout.write('\rstatus: %d%%' % percent_complete)
    sys.stdout.flush()

    # NOTES:

    # Deleting an index:
    #   es_client.indices.delete(index = index_name, ignore=[400, 404])

    # Creating an index:
    #   es_client.indices.create(index = index_name)
