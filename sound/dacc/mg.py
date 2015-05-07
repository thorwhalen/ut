__author__ = 'thor'

import os
import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor
from itertools import imap

from ut.sound import util as sutil
from ut.daf.manip import reorder_columns_as
from ut.sound.util import Sound
from ut.pstr.trans import str_to_utf8_or_bust


class MgDacc(object):
    def __init__(self, db, collection, root_folder, path_field='_id', mg_client_kwargs={}):
        self.mgc = MongoClient(**mg_client_kwargs)[db][collection]
        self.root_folder = root_folder
        self.path_field = path_field

    def filepath_of(self, path):
        return str_to_utf8_or_bust(os.path.join(self.root_folder, path))

    def get_wf_and_sr(self, path, **kwargs):
        return sutil.wf_and_sr_from_filepath(self.filepath_of(path), **kwargs)

    def get_sound(self, path_or_doc, **kwargs):
        if not isinstance(path_or_doc, basestring):
            path_or_doc = path_or_doc.copy()
            file_path = path_or_doc.pop(self.path_field)
            kwargs = dict(kwargs, **path_or_doc)
            path_or_doc = file_path
        name = kwargs.pop('name', os.path.splitext(os.path.basename(path_or_doc))[0])
        try:
            wf, sr = self.get_wf_and_sr(path_or_doc, **kwargs)
        except TypeError:
            kwargs.pop('channels')
            kwargs.pop('frames')
            wf, sr = self.get_wf_and_sr(path_or_doc, **kwargs)
        return Sound(wf=wf, sr=sr, name=name)

    def get_sound_iterator(self, find_args={}, find_kwargs={}):
        """
        Util to flip through sounds.
        You can do, for example:
            sound_iterator = self.get_sound_iterator
        and then run the following several times:
            sound = sound_iterator.next(); sound.display_sound()
        """
        if not find_args and not find_kwargs:
            cursor = self.mgc.find()
        else:
            cursor = self.mgc.find(*find_args, **find_kwargs)
        return imap(lambda x: self.get_sound(path_or_doc=x[self.path_field]), cursor)


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
            for seg in ci[self.segment_field]:
                dd = {'path': ci[self.path_field]}
                for tag_key in kv_tag_keys:
                    dd.update({tag_key: ci[self.kv_tag_field].get(tag_key, None)})
                dd.update(seg['fv'])
                dd.update({'offset_s': seg['offset_s'], 'duration': seg['duration']})
                d += [dd]
        d = reorder_columns_as(pd.DataFrame(d), ['path'] + kv_tag_keys + ['offset_s', 'duration'])
        return d

    # def get_sound(self, *args, **kwargs):
    #     # if len(args) > 0:
    #     #     kwargs['path_or_doc'] = args[0]
    #     return super(SegmentDacc, self).get_sound(path_or_doc=, **kwargs)

            # return super(SegmentDacc, self).get_sound(args[0], **kwargs)
        # return super(SegmentDacc, self).get_sound(path_or_doc=kwargs['path'],
        #                                           offset_s=kwargs['offset_s'],
        #                                           duration=kwargs['duration'])

    def get_segment_iterator(self, only_segments=True, fields=None, *args, **kwargs):
        cursor = self.mgc.find(*args, **kwargs)

        def segment_iterator():
            for d in cursor:
                segments = d.pop(self.segment_field)
                if segments is not None:
                    for dd in segments:
                        if not only_segments:
                            dd = dict(d, **dd)
                        if fields is None:
                            yield dd
                        else:
                            yield {k: v for k, v in dd.iteritems() if k in fields}

        return segment_iterator()

    def get_sound_iterator(self, *args, **kwargs):
        """
        Util to flip through sounds.
        You can do, for example:
            sound_iterator = self.get_sound_iterator
        and then run the following several times:
            sound = sound_iterator.next(); sound.display_sound()
        """

        cursor = self.mgc.find(*args, **kwargs)
        return imap(self.get_sound, cursor)
