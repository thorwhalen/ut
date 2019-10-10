__author__ = 'thor'
"""
Based on code from Julien Palard.
"""


from argparse import ArgumentParser, FileType
import json
import requests
from time import sleep


class ElasticSearch():
    def __init__(self, url):
        self.url = url

    def request(self, method, path, data=None):
        return (requests.request(
                method, 'http://%s/%s' % (self.url, path),
                data=data,
                headers={'Content-type': 'application/json'}).json())

    def post(self, path, data):
        return self.request('post', path, data)

    def get(self, path, data=None):
        return self.request('get', path, data)

    def scan_and_scroll(self, index):
        response = self.get('%s/_search?search_type=scan&scroll=1m' % index,
                            data=json.dumps({"query": {"match_all": {}},
                                             "size": 100}))
        while True:
            response = self.get('_search/scroll?scroll=1m',
                                data=response['_scroll_id'])
            if len(response['hits']['hits']) == 0:
                return
            yield response['hits']['hits']

    def set_mapping(self, index, mappings):
        return self.post(index, data=json.dumps(mappings))

    def count(self, index):
        response = self.get('%s/_search' % index)
        return response['hits']['total'] if 'hits' in response else 0

    def bulk_insert(self, index, bulk):
        return self.post('_bulk',
                         data=''.join(
                         json.dumps({'create': {'_index': index,
                                                '_type': line['_type']}}) +
                         "\n" +
                         json.dumps(line['_source']) + "\n" for line in bulk))

    def drop(self, index):
        return self.request('delete', index)

    def alias(self, index, to):
        return self.request('put', '%s/_alias/%s' % (index, to))


def change_mapping_and_reindex(elasticsearch, mapping_file, index):
    es = ElasticSearch(elasticsearch)

    mapping_text = mapping_file.read()
    temporary_index = None
    for i in range(10):
        try_temporary_index = index + '-tmp-' + str(i)
        print("Setting mapping to %s" % try_temporary_index)
        response = es.set_mapping(try_temporary_index,
                                  json.loads(mapping_text))
        if 'acknowledged' in response and response['acknowledged']:
            temporary_index = try_temporary_index
            break
    if temporary_index is None:
        print("Can't find a temporary index to work with.")
        return False

    old_index_count = es.count(index)
    new_index_count = es.count(temporary_index)

    for bulk in es.scan_and_scroll(index):
        es.bulk_insert(temporary_index, bulk)
        new_index_count = es.count(temporary_index)
        percent = 100 * new_index_count / old_index_count
        print(("\r%.2f%%" + 10 * " ") % percent, end=' ')
    print("\nDone")

    for i in range(100):
        new_index_count = es.count(temporary_index)
        if new_index_count == old_index_count:
            print("OK, same number of raws in both index.")
            break
        print(("Not the same number of raws in old and new... "
               "waiting a bit..."
               "(old=%d, new=%d)" % (old_index_count, new_index_count)))
        if i > 10:
            print(("Oh fsck, not the same number of raws in old and new... "
                   "aborting."
                   "(old=%d, new=%d)" % (old_index_count, new_index_count)))
            return
        sleep(1)

    print("Deleting %s" % index)
    es.drop(index)
    print("Aliasing %s to %s" % (temporary_index, index))
    es.alias(temporary_index, index)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Remap and reindex the given index, but only if you stoped"
        "writing to it (will fail if you're writing")
    parser.add_argument('--index', help='index to remap')
    parser.add_argument('--elasticsearch', help='ES host')
    parser.add_argument('--mapping',
                        help='Mapping file, starts with {"mappings"...',
                        type=FileType('r'))
    args = parser.parse_args()
    change_mapping_and_reindex(args.elasticsearch, args.mapping, args.index)