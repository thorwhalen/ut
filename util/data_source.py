__author__ = 'thorwhalen'

from pfile.accessor import Accessor
import os
from os import environ
import pfile.accessor as pfile_accessor
import pickle
import pfile.to as pfile_to

MS_DATA = getenv('MS_DATA')
LOCATION_LOCAL = 'LOCAL'
LOCATION_S3 = 'S3'

########################################################################################################################
# FACTORIES

def for_global_local():
    mother_root = MS_DATA
    dir_dict = {
                'root': '',
                'venere': 'venere',
                'log': 'log',
                'hotel': 'venere/hotel',
                'geography': 'venere/geography',
                'proptype': 'venere/property_type',
                'khan_html': 's3/ut-slurps/html/', # 'html/google_results_tests/'
                'khan_parsed_results': 's3/ut-slurps/parsed', # 'dict/google_results_parse_result'
                'khan_info_dict': 's3/semantics-data/gresult_info_dict', # 'dict/gresult_trinity_info'
            }
    a_dict = {
                'root': '',
                'venere': 'venere',
                'hotel': 'venere/hotel',
                'geography': 'venere/geography',
                'proptype': 'venere/property_type'
            }
    f_dict = {
        'khan_term_map': os.path.join(dir_dict['venere'],'term_map.dict'),
        'khan_parse_search_terms_log': os.path.join(dir_dict['log'],'khan_parse_search_terms.log')
    }
    d_dict = {
        'khan_term_map': os.path.join(f_dict['khan_term_map'])
    }
    instance = DataSource(
        location=LOCATION_LOCAL,
        mother_root=mother_root,
        dir_dict=dir_dict,
        a_dict=a_dict,
        f_dict=f_dict,
        d_dict=d_dict
    )
    return instance


def for_basic_local():
    mother_root = MS_DATA
    dir_dict = {
                'root': '',
                'venere': 'venere',
                'hotel': 'venere/hotel',
                'geography': 'venere/geography',
                'proptype': 'venere/property_type'
            }
    a_dict = {
                'root': '',
                'venere': 'venere',
                'hotel': 'venere/hotel',
                'geography': 'venere/geography',
                'proptype': 'venere/property_type'
            }
    f_dict = {
        'term_mapping.dict': os.path.join(dir_dict['venere'],'term_mapping.dict')
    }
    instance = DataSource(
        location=LOCATION_LOCAL,
        mother_root=mother_root,
        dir_dict=dir_dict,
        a_dict=a_dict,
        f_dict=f_dict
    )
    return instance



########################################################################################################################
# CLASSES

class DataSource(object):
    def __init__(self,
                 location=LOCATION_LOCAL,
                 mother_root='',
                 dir_dict=None,
                 a_dict=None,
                 f_dict=None,
                 d_dict=None):
        self.location=location
        # self.dacc = Accessor(location=location)
        # self.dir_root = self.dacc.filepath('')
        a_dict = a_dict or dir_dict
        if mother_root:
            if mother_root[-1]!='/':
                mother_root = mother_root + '/'
        self.dir = Directories(
            mother_root=mother_root,
            dir_dict=dir_dict)
        self.a = Accessors(location=location, a_dict=a_dict, mother_root=mother_root)
        self.f = Files(f_dict, mother_root=mother_root)
        self.d = Data(d_dict, mother_root=mother_root)

    def check(self):
        if self.location==LOCATION_LOCAL:
        # check existence of .dir directories
            for k,v in list(self.dir.__dict__.items()):
                if not os.path.exists(v):
                    print("!!!Non existent directory: %s" % v)
            # check existence of root directories of .a (accessors)
            for k,v in list(self.a.__dict__.items()):
                root_directory = v.filepath('')
                if not os.path.exists(root_directory):
                    print("!!!Non existent accessor root directory: %s" % root_directory)
            # check existence of files of .f
            for k,v in list(self.f.__dict__.items()):
                if not os.path.exists(v):
                    print("!!!Non existent file: %s" % v)
        elif self.location==LOCATION_S3:
            raise ValueError("Don't know how to check S3 sources yet")

    def print_info(self, **kwargs):
        print("---------------------------------")
        print("-------  DataSource Info --------")
        print("")
        print("Location: %s" % self.location)
        print("")
        print("------- .dir (directories) ------")
        for k,v in list(self.dir.__dict__.items()):
                print("  %s: %s" % (k, v))
        print("")
        print("------- .a (accessors) ----------")
        for k,v in list(self.a.__dict__.items()):
                print("  %s: %s" % (k, v.filepath('')))
        print("")
        print("------- .f (files) --------------")
        for k,v in list(self.f.__dict__.items()):
                print("  %s: %s" % (k, v))
        if 'print_data' not in kwargs or kwargs['print_data']==True:
            print("")
            print("------- .d (data) ---------------")
            for k,v in list(self.d.__dict__.items()):
                    print("  %s: %s" % (k, v))

class Directories(object):
    def __init__(self,
                 mother_root=None,
                 dir_dict=None):
        if mother_root[-1]!='/':
            mother_root = mother_root + '/'
        for k,v in list(dir_dict.items()):
            self.__setattr__(k,mother_root+v)

class Accessors(object):
    def __init__(self, location, a_dict, mother_root=''):
        if location==LOCATION_LOCAL:
            for k,v in list(a_dict.items()):
                if isinstance(v, str):
                    # if v is simply a string, take it as the directory relative_root of the accessor
                    self.__setattr__(k,pfile_accessor.for_local(relative_root=os.path.join(mother_root,v)))
                elif isinstance(v, dict):
                    # if it's a dict, take it as a kwargs of the accessor construction
                    self.__setattr__(k,pfile_accessor.for_local(**v))

class Files(object):
    def __init__(self, f_dict, mother_root=''):
        for k,v in list(f_dict.items()):
            self.__setattr__(k,mother_root+v)

class Data(object):
    def __init__(self, d_dict, load_if_existing_file=True, mother_root=''):
        if d_dict:
            for k, v in list(d_dict.items()):
                if isinstance(v, str) and load_if_existing_file:
                    # if v (or mother_root+v) exists as a file, load it in v
                    v_path = ''
                    if os.path.exists(v):
                        v_path = v
                    elif os.path.exists(mother_root+v):
                        v_path = mother_root+v
                    if v_path:
                        try:
                            # try to unpickle
                            v = pickle.load(open(v_path,'r'))
                        except:
                            # load as a string
                            v = pfile_to.string(v_path)
                self.__setattr__(k,v)
