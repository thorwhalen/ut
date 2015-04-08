__author__ = 'thor'

import os
import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor

from ut.sound import util as sutil
from ut.daf.manip import reorder_columns_as


class MgDacc(object):
    def __init__(self, db, collection, root_folder, path_field='_id', mg_client_kwargs={}):
        self.mgc = MongoClient(**mg_client_kwargs)[db][collection]
        self.root_folder = root_folder
        self.path_field = path_field

    def filepath_of(self, path):
        return os.path.join(self.root_folder, path)

    def get_wf_and_sr(self, path, **kwargs):
        return sutil.wf_and_sr_from_filepath(self.filepath_of(path), **kwargs)


class SegmentDacc(MgDacc):
    def __init__(self, db, collection, root_folder, path_field='_id', mg_client_kwargs={},
                 segment_field='segments', feat_field='fv', tag_field='tags', kv_tag_field='kv_tags'):
        super(SegmentDacc, self).__init__(db, collection, root_folder, path_field, mg_client_kwargs)
        self.segment_field = segment_field
        self.feat_field = feat_field
        self.tag_field = tag_field
        self.kv_tag_field = kv_tag_field

    def get_data_with_tags(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], Cursor):
            c = args[0]
        else:
            c = self.mgc.find(*args, **kwargs)
        d = list()
        for ci in c:
            for seg in ci['segments']:
                dd = {'path': ci[self.path_field], 'tags': ci[self.tag_field]}
                dd.update(seg['fv'])
                dd.update({'offset_s': seg['offset_s'], 'duration': seg['duration']})
                d += [dd]
        d = reorder_columns_as(pd.DataFrame(d), ['path', 'tags', 'offset_s', 'duration'])
        return d

    def get_data_with_kv_tags(self, *args, **kwargs):
        if 'kv_tag_keys' in kwargs.keys():
            kv_tag_keys = kwargs.get('kv_tag_keys')
            kwargs.pop('kv_tag_keys')
        else:
            kv_tag_keys = ['move_direction', 'vehicle_type']

        if len(args) > 0 and isinstance(args[0], Cursor):
            c = args[0]
        else:
            c = self.mgc.find(*args, **kwargs)
        d = list()
        for ci in c:
            for seg in ci['segments']:
                dd = {'path': ci[self.path_field]}
                for tag_key in kv_tag_keys:
                    dd.update({tag_key: ci[self.kv_tag_field].get(tag_key, None)})
                dd.update(seg['fv'])
                dd.update({'offset_s': seg['offset_s'], 'duration': seg['duration']})
                d += [dd]
        d = reorder_columns_as(pd.DataFrame(d), ['path'] + kv_tag_keys + ['offset_s', 'duration'])
        return d


