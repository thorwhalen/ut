"""File access utils"""
__author__ = 'thorwhalen'

# from ut.datapath import datapath
import pickle
import os
from ut.util.importing import get_environment_variable
import pandas as pd
import ut.pfile.to as file_to
import ut.pfile.name as pfile_name
import ut.pstr.to as pstr_to
from ut.serialize.local import Local
from ut.serialize.s3 import S3
from os import (
    environ,
)  # does this load the whole array? Can we just take MS_DATA instead?
import ut.pstr.trans as pstr_trans
import shutil

try:
    MS_DATA = get_environment_variable('MS_DATA')
except KeyError:
    MS_DATA = ''


LOCATION_LOCAL = 'LOCAL'
LOCATION_S3 = 'S3'

####################################################################################################################
# Quick Utils
def ms_data_path(relative_root, root_folder=MS_DATA):
    return os.path.join(pfile_name.ensure_slash_suffix(root_folder), relative_root)


####################################################################################################################
# FACTORIES
def for_local(
    relative_root='',
    read_only=False,
    extension=None,
    force_extension=False,
    root_folder=MS_DATA,
    **kwargs
):
    # if a full path (i.e. starting with "/" is entered as a relative_root, then take it as the sound_file_root_folder
    if relative_root and ((relative_root[0] == '/') or (relative_root[0] == '~')):
        root_folder = relative_root
        relative_root = ''
    elif relative_root == 'test':  # if relative root is test...
        relative_root = 'test'
        print(
            'you asked for a local test, so I forced the root to be %s' % relative_root
        )
    # ensure that sound_file_root_folder ends with a "/"
    file_handler = FilepathHandler(
        relative_root=pfile_name.ensure_slash_suffix(root_folder) + relative_root
    )
    # take care of extensions
    if extension:
        extension_handler = ExtensionHandler(
            extension=extension, force_extension=force_extension
        )
        file_loc_proc = lambda x: file_handler.process(extension_handler.process(x))
    else:
        file_loc_proc = file_handler.process
    instance = Accessor(
        relative_root=relative_root,
        extension=extension,
        force_extension=force_extension,
        file_loc_proc=file_loc_proc,
        location=LOCATION_LOCAL,
        read_only=read_only,
        **kwargs
    )
    instance._set_local_defaults()
    return instance


def for_s3(
    relative_root='loc-data',
    read_only=False,
    extension=None,
    force_extension=False,
    **kwargs
):
    if relative_root == 'test':
        relative_root = 'loc-data/test'
        print('you asked for a s3 test, so I forced the root to be %s' % relative_root)
    file_handler = FilepathHandler(relative_root=relative_root)
    if extension:
        extension_handler = ExtensionHandler(
            extension=extension, force_extension=force_extension
        )
        file_loc_proc = lambda x: file_handler.process(extension_handler.process(x))
    else:
        file_loc_proc = file_handler.process
    instance = Accessor(
        relative_root=relative_root,
        extension=extension,
        force_extension=force_extension,
        file_loc_proc=file_loc_proc,
        location=LOCATION_S3,
        read_only=read_only,
        **kwargs
    )
    save_kwargs = instance.mk_save_kwargs(relative_root)
    try:
        bucket_name = save_kwargs['bucket_name']
        base_folder = save_kwargs['key_name']
    except:
        print("couldn't get bucket_name and key_name for relative_root")
    instance.s3 = S3(bucket_name=bucket_name, base_folder=base_folder)
    instance._set_s3_defaults()
    return instance


####################################################################################################################


class Accessor(object):
    LOCATION_LOCAL = LOCATION_LOCAL
    LOCATION_S3 = LOCATION_S3

    def __init__(
        self,
        file_loc_proc=None,
        location=LOCATION_LOCAL,
        mk_save_kwargs=None,
        pre_save_proc=None,
        save_fun=None,
        mk_load_kwargs=None,
        load_fun=None,
        post_load_proc=None,
        read_only=False,
        **kwargs
    ):
        # if file_loc_proc:
        #     self.file_loc_proc = file_loc_proc
        # else:
        #     self.file_loc_proc = FilepathHandler().process
        self.file_loc_proc = file_loc_proc
        self.location = location

        self.mk_save_kwargs = mk_save_kwargs
        self.pre_save_proc = pre_save_proc
        self.save_fun = save_fun

        self.mk_load_kwargs = mk_load_kwargs
        self.load_fun = load_fun
        self.post_load_proc = post_load_proc

        self.read_only = read_only

        for k, v in list(kwargs.items()):
            self.__setattr__(k, v)

        self._guess_missing_attributes()

    def __call__(self, *args, **kwargs):
        return self.filepath(*args, **kwargs)

    ####################################################################################################################
    # INSTANCE METHODS

    def root_folder(self):
        if self.extension:
            return self.file_loc_proc('')[: (-len(self.extension))]
        else:
            return self.file_loc_proc('')

    def filepath(self, file_spec):
        return self.file_loc_proc(file_spec)

    def exists(self, file_spec):
        return os.path.exists(self.filepath(file_spec))

    def save(self, obj, file_spec, **kwargs):
        if self.read_only:
            raise BaseException("read_only was set to True, so you can't save anything")
        else:
            # make the dict specifying the input to the save_fun
            file_spec = self.file_loc_proc(file_spec)
            if self.pre_save_proc:
                obj = self.pre_save_proc(obj)
            if self.mk_save_kwargs:
                file_spec_kwargs = self.mk_save_kwargs(file_spec)
                self.save_fun(obj, **file_spec_kwargs)
            else:
                self.save_fun(obj, file_spec)

    def append(self, obj, file_spec, **kwargs):  # TODO: Write this code someday
        """
        Intent of this function is to append data to a file's data without having to specify how to do so.
        For example, if the obj is a string and the file is a text file, use file append.
        If obj is a pickled dataframe, the effect (however you do it--hopefully there's a better way than loading the
        data, appending, and saving the final result) should be to have a pickled version of the old and new dataframes
        appended.
        Etc.
        """
        pass
        # if isinstance(obj, basestring):
        #     raise ValueError("strings not implemented yet")
        # elif isinstance(obj, (pd.DataFrame, pd.Series)):
        #     pass

    def load(self, file_spec, **kwargs):
        file_spec = self.file_loc_proc(file_spec)
        if pfile_name.get_extension(file_spec) not in ['.xls', '.xlsx']:
            if self.mk_load_kwargs:
                file_spec_kwargs = self.mk_load_kwargs(file_spec)
                obj = self.load_fun(**file_spec_kwargs)
            else:
                obj = self.load_fun(file_spec)
            if self.post_load_proc:
                obj = self.post_load_proc(obj)
        else:
            # obj = pd.read_excel(file_spec,  **kwargs)
            xls = pd.ExcelFile(file_spec)
            kwargs = dict(
                {'sheetname': xls.sheet_names[0]}, **kwargs
            )  # take first sheet if sheet not specified
            obj = pd.read_excel(file_spec, **kwargs)
            # obj = xls.parse(**kwargs)
        return obj

    def copy_local_file_to(self, local_file_path, target_file_spec):
        """
        Copies a file from the local computer to self.filepath(target_file_spec)
        :param local_file_path:
        :param target_file_spec:
        :return:
        """
        if self.read_only:
            raise BaseException(
                "read_only was set to True, so you can't copy anything to this location"
            )
        else:
            if self.location == LOCATION_LOCAL:
                if not os.path.exists(local_file_path):
                    local_file_path = self.filepath(local_file_path)
                shutil.copyfile(local_file_path, self.filepath(target_file_spec))
            elif self.location == LOCATION_S3:
                # make the dict specifying the input to the save_fun
                target_file_spec = self.file_loc_proc(target_file_spec)
                if self.pre_save_proc:
                    local_file_path = self.pre_save_proc(local_file_path)
                if self.mk_save_kwargs:
                    file_spec_kwargs = self.mk_save_kwargs(target_file_spec)
                    self.copy_local_file_to_fun(local_file_path, **file_spec_kwargs)
                else:
                    raise ("this shouldn't happen")
            else:
                raise ValueError('unknown location')

    def copy_to(self, target_relative_root, file_spec, target_location=None):
        if isinstance(target_relative_root, str):
            (
                target_relative_root,
                target_location,
            ) = _make_a_file_loc_proc_and_location_from_string_specifications(
                target_relative_root, target_location
            )
            # make a file accessor for the (target_location, target_relative_root)
            facc = Accessor(
                relative_root=target_relative_root, location=target_location
            )

    ####################################################################################################################
    # PARTIAL FACTORIES

    def _add_extension_handler(self, extension, force_extension=False):
        extension_handler = ExtensionHandler(
            extension=extension, force_extension=force_extension
        )
        self.file_loc_proc = lambda x: self.file_loc_proc(extension_handler.process(x))

    def _guess_missing_attributes(self):
        if self.file_loc_proc is None:  # if no file_loc_proc is given
            if self.location is not None and isinstance(self.location, str):
                self.file_loc_proc == self.location
            else:
                self.file_loc_proc == LOCATION_LOCAL
        elif isinstance(self.file_loc_proc, str):  # if file_loc_proc is a string
            (
                self.file_loc_proc,
                self.location,
            ) = _make_a_file_loc_proc_and_location_from_string_specifications(
                self.file_loc_proc, self.location
            )
            # if self.file_loc_proc==LOCATION_LOCAL:
            #     self.location = LOCATION_LOCAL
            #     self.file_loc_proc = ''
            # elif self.file_loc_proc==LOCATION_S3:
            #     self.location = LOCATION_S3
            #     self.file_loc_proc = ''
            # else:
            #     if self.location==LOCATION_LOCAL:
            #         self.file_loc_proc = FilepathHandler(relative_root=os.path.join(MS_DATA,self.file_loc_proc)).process
            #     elif self.location==LOCATION_S3:
            #         self.file_loc_proc = FilepathHandler(relative_root=os.path.join('loc-data',self.file_loc_proc)).process
        # set defaults for remaining missing attributes
        self._set_defaults()

    def _set_defaults(self):
        if self.location is None:
            print("setting location to LOCAL (because you didn't specify a location)")
            self.location = LOCATION_LOCAL
        if self.location == LOCATION_LOCAL:
            self._set_local_defaults()
        elif self.location == LOCATION_S3:
            self._set_s3_defaults()

    def _set_local_defaults(self, root_folder=MS_DATA):
        # set defaults for local if attr is None
        self.file_loc_proc = (
            self.file_loc_proc
            or FilepathHandler(relative_root=os.path.join(root_folder)).process
        )
        self.save_fun = self.save_fun or LocalIOMethods().unicode_save
        self.load_fun = self.load_fun or LocalIOMethods().unicode_load
        # self.pre_save_proc = self.pre_save_proc or FilepathHandler().process
        # self.post_load_proc = self.post_load_proc or FilepathHandler().process

    def _set_s3_defaults(self):
        # set defaults for local if attr is None
        self.file_loc_proc = (
            self.file_loc_proc or FilepathHandler(relative_root='loc-data').process
        )
        self.mk_save_kwargs = fullpath_to_s3_kargs
        self.mk_load_kwargs = fullpath_to_s3_kargs
        self.save_fun = self.save_fun or S3IOMethods().unicode_save
        self.load_fun = self.load_fun or S3IOMethods().unicode_load
        self.copy_local_file_to_fun = S3IOMethods().copy_local_file_to_fun

    ####################################################################################################################
    # OBJECT UTILS

    def local_file_loc_proc_simple(self, file_spec):
        # add extension
        file_spec = self.handle_extension(file_spec)
        # remove slash suffix if present (because self.sound_file_root_folder ends with / already)
        if file_spec.startswith('/'):
            file_spec = file_spec[1:]

    def handle_extension(self, file_spec):
        if self.extension:
            if self.force_extension:
                file_spec = pfile_name.replace_extension(file_spec, self.extension)
            else:
                file_spec = pfile_name.add_extension_if_not_present(
                    file_spec, self.extension
                )
        return os.path.join(self.root_folder, file_spec)


####################################################################################################################
# OTHER UTILS


def _make_a_file_loc_proc_and_location_from_string_specifications(
    file_loc_proc, location
):
    if file_loc_proc is None and isinstance(location, str):
        file_loc_proc = location + '/'
        location = None
    elif location is None and isinstance(file_loc_proc, str):
        first_folder = pfile_name.get_highest_level_folder(location)
        if first_folder in [LOCATION_LOCAL, LOCATION_S3]:
            location = first_folder  # set the location to first_folder
            file_loc_proc.replace(location + '/', '')  # remove the first_folder
        else:
            raise ValueError(
                "location was not specified and couldn't be guessed from the file_loc_proc"
            )
    else:
        raise ValueError(
            "you've neither specified a file_loc_proc (as a file_loc_proc) nor a location"
        )
        # make a file accessor for the (location, target_relative_root)
    file_loc_proc = FilepathHandler(
        relative_root=os.path.join(location, file_loc_proc)
    ).process
    return (file_loc_proc, location)


def file_loc_proc_from_full_path(fullpath):
    return FilepathHandler(relative_root=fullpath).process


def fullpath_to_s3_kargs(filename):
    # remove slash suffix if present (because self.sound_file_root_folder ends with / already)
    if filename.startswith('/'):
        filename = filename[1:]
    mother_root = pfile_name.get_highest_level_folder(filename)
    rest_of_the_filepath = filename.replace(mother_root + '/', '', 1)
    return {'bucket_name': mother_root, 'key_name': rest_of_the_filepath}


class ExtensionHandler(object):
    def __init__(self, extension=None, force_extension=False):
        self.extension = extension
        self.force_extension = force_extension

    def process(self, file_spec):
        if self.force_extension:
            return pfile_name.replace_extension(file_spec, self.extension)
        else:
            return pfile_name.add_extension_if_not_present(file_spec, self.extension)


class FilepathHandler(object):
    def __init__(self, relative_root=''):
        self.relative_root = relative_root

    def process(self, filepath=''):
        return os.path.join(self.relative_root, filepath)


##### LOCAL METHODS


class LocalIOMethods(object):
    def __init__(self, encoding='UTF-8'):
        self.encoding = encoding

    def unicode_save(self, obj, filepath=None, **kwargs):
        if isinstance(obj, str):
            # pstr_to.file(string=pstr_trans.to_unicode_or_bust(obj), tofile=filepath, encoding=self.encoding)
            # pstr_to.file(string=pstr_trans.to_utf8_or_bust_iter(obj), tofile=filepath, encoding=self.encoding)
            # pstr_to.file(string=pstr_trans.str_to_utf8_or_bust(obj), tofile=filepath, encoding=self.encoding)
            pstr_to.file(string=obj, tofile=filepath, encoding=self.encoding)
        else:
            pickle.dump(obj=obj, file=open(filepath, 'w'))

    def simple_save(self, obj, filepath=None, **kwargs):
        if isinstance(obj, str):
            pstr_to.file(string=obj, tofile=filepath, encoding=self.encoding)
        else:
            pickle.dump(obj=obj, file=open(filepath, 'w'))

    def unicode_load(self, filepath=None, **kwargs):
        """
        try pd.from_pickle, then pickle.loading, and if it doesn't work, try file_to.string
        """
        return pstr_trans.to_unicode_or_bust(
            self.simple_load(filepath=filepath, **kwargs)
        )
        # try:
        #     try:  # getting it as a pandas object
        #         return pstr_trans.to_unicode_or_bust(pd.read_pickle(path=filepath))
        #     except Exception:  # getting it as a pickled object
        #         return pstr_trans.to_unicode_or_bust(pickle.load(file=open(filepath, 'r')))
        # except Exception:  # getting it as a string
        #     return pstr_trans.to_unicode_or_bust(file_to.string(filename=filepath))

    def simple_load(self, filepath=None, **kwargs):
        """
        try pd.read_pickle, pickle.load, and file_to.string in that order
        """
        try:
            try:  # getting it as a pandas object
                return pd.read_pickle(path=filepath)
            except Exception:  # getting it as a pickled object
                return pickle.load(file=open(filepath, 'r'))
        except Exception:  # getting it as a string
            return file_to.string(filename=filepath)


##### S3 METHODS


class S3IOMethods(object):
    def __init__(self, **kwargs):
        self.s3 = S3(**kwargs)

    def unicode_save(self, obj, key_name, bucket_name):
        if isinstance(obj, str):
            self.s3.dumps(
                the_str=pstr_trans.to_unicode_or_bust(obj),
                key_name=key_name,
                bucket_name=bucket_name,
            )
        else:
            self.s3.dumpo(obj=obj, key_name=key_name, bucket_name=bucket_name)

    def simple_save(self, obj, key_name, bucket_name):
        if isinstance(obj, str):
            self.s3.dumps(the_str=obj, key_name=key_name, bucket_name=bucket_name)
        else:
            self.s3.dumpo(obj=obj, key_name=key_name, bucket_name=bucket_name)

    def unicode_load(self, key_name, bucket_name):
        """
        try pickle.loading, and if it doesn't work, try file_to.string
        """
        try:
            return self.s3.loado(key_name=key_name, bucket_name=bucket_name)
        except:
            return pstr_trans.to_unicode_or_bust(
                self.s3.loads(key_name=key_name, bucket_name=bucket_name)
            )

    def simple_load(self, key_name, bucket_name):
        """
        try pickle.loading, and if it doesn't work, try file_to.string
        """
        try:
            return self.s3.loado(key_name=key_name, bucket_name=bucket_name)
        except:
            return self.s3.loads(key_name=key_name, bucket_name=bucket_name)

    def copy_local_file_to_fun(self, filepath, key_name, bucket_name):
        return self.s3.dumpf(f=filepath, key_name=key_name, bucket_name=bucket_name)
