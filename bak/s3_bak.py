from boto.s3.connection import S3Connection
from boto.s3.key import Key
from boto.exception import S3CopyError, S3ResponseError
import tempfile
import pickle
import os
import pfile.name as pfile_name
import serialize.utils as serialize_utils

# TODO ! MJM - I am updating params that take bucket to now take bucket_name instead since that is a more likely
# use case. I need to check all code that uses these methods, as well as the test notebook.


class S3(object):
    """
    Interaction with Amazon S3.
    """

    def __init__(self, bucket_name, base_folder=None, extension=None, force_extension=False, encoding=None, access_key=None, secret=None):
        """
        Creates an instance with a handle on the S3 bucket corresponding to bucket_name.

        If access_key and/or secret are not passed in, assumes we are accessing erenev's aws account and that the
        access info is stored as environment variables on the current server.

        Connection and bucket are available to clients via self properties, in case clients wish to use those objects
        directly.
        """
        access_key = access_key or os.getenv('VEN_S3_ACCESS_KEY')
        secret = secret or os.getenv('VEN_S3_SECRET')

        # note - this calls the setter
        self.base_folder = base_folder
        self.extension = extension
        self.force_extension = force_extension
        self.encoding = encoding or 'UTF-8'

        self.connection = S3Connection(access_key, secret, host='s3-eu-west-1.amazonaws.com')
        self.bucket = self.connection.get_bucket(bucket_name)

    @property
    def base_folder(self):
        return self._base_folder

    @base_folder.setter
    def base_folder(self, base_folder):
        self._base_folder = self._ensure_good_folder_name(base_folder)

    def filepath(self, filename=None):
        if not filename:
            return self.root_folder

        # remove slash suffix if present (because self.sound_file_root_folder ends with / already)
        if filename.startswith('/'):
            filename = filename[1:]
        # add extension
        if self.extension:
            if self.force_extension:
                filename = pfile_name.replace_extension(filename, self.extension)
            else:
                filename = pfile_name.add_extension_if_not_present(filename, self.extension)
        return os.path.join(self.base_folder, filename)


    def dumpo(self, obj, key_name, folder=None, bucket_name=''):
        """
        --
        For saving objects to S3
        Do not use this method for saving strings. It will work, but using dumps() is more efficient.
        --
        params:
          obj: the object to save
          key_name: the value that will be assigned to key.key. May be full path, including folders.
          folder: optional - the folder name(s). Only use this param if you did not use full path in key_name
          bucket_name: optional - overrides the default bucket name passed in during init.
        return:
          number of bytes
        """
        bucket = self._get_new_bucket_or_default(bucket_name)
        s3_key = self._get_s3_key_for_dump(self.filepath(key_name), folder, bucket)

        with tempfile.TemporaryFile() as tempf:
            pickle.dump(obj, tempf)
            tempf.seek(0)
            return s3_key.set_contents_from_file(tempf)

    def dumps(self, the_str, key_name, folder=None, bucket_name=None, encoding=None):
        """
        --
        For saving strings to S3
        --
        params:
          the_str: the string to save
          key_name: the value that will be assigned to key.key. May be full path, including folders.
          folder: optional - the folder name(s). Only use this param if you did not use full path in key_name
          bucket: optional - overrides the default bucket name passed in during init.
        return:
          number of bytes
        """
        assert isinstance(the_str,
                          str), 'the_str must be an instance of basestring, but was an instance of {}'.format(type(the_str))
        encoding = encoding or self.encoding
        bucket = self._get_new_bucket_or_default(bucket_name)
        s3_key = self._get_s3_key_for_dump(self.filepath(key_name), folder, bucket)
        return str(s3_key.set_contents_from_string(the_str),encoding)

    def loado(self, key_name, folder=None, bucket_name='', deserialize_f=lambda x: pickle.load(x)):
        """
        --
        For loading objects from S3
        Do not use this method for loading strings since de-pickling will throw an error.
        --
        params:
          key_name: the value that will be assigned to key.key. May be full path, including folders.
          folder: optional - the folder name(s). Only use this param if you did not use full path in key_name
          bucket: optional - overrides the default bucket name passed in during init.
        return:
          the serialized object
        """
        bucket = self._get_new_bucket_or_default(bucket_name)
        s3_key = self._get_s3_key_for_load(self.filepath(key_name), folder, bucket)
        with tempfile.TemporaryFile() as tempf:
            s3_key.get_contents_to_file(tempf)
            tempf.seek(0)
            val = deserialize_f(tempf)
            return val

    def loads(self, key_name, folder=None, bucket_name=None, encoding=None):
        """
        --
        For loading strings from S3
        --
        params:
          key_name: the value that will be assigned to key.key. May be full path, including folders.
          folder: optional - the folder name(s). Only use this param if you did not use full path in key_name
          bucket: optional - overrides the default bucket name passed in during init.
        return:
          the serialized string
        """
        encoding = encoding or self.encoding
        bucket = self._get_new_bucket_or_default(bucket_name)
        s3_key = self._get_s3_key_for_load(self.filepath(key_name), folder, bucket)
        return str(s3_key.get_contents_as_string(),encoding)

    def get_all_keys(self, folder, clean=True, bucket_name=''):
        """
        Returns a generator of all keys found in the specified folder and (optional=default) bucket.
        If clean = True (default), the empty 'folder' key that is returned by S3 API is removed and
        the key names have the folder name removed from them.
        If clean = False, the results are returned as-is from bucket.list for the folder.
        """
        bucket = self._get_new_bucket_or_default(bucket_name)

        if not folder.endswith('/'):
            folder += '/'

        key_result_set = bucket.list(prefix=folder)

        if clean:
            key_result_set_no_empty = (k for k in key_result_set if k.name.replace(folder, '') != '')
            key_result_set_no_folder_names = (self._remove_folder_from_name(k, folder) for k in key_result_set_no_empty)
            return key_result_set_no_folder_names
        else:
            return key_result_set

    def update_metadata(self, metadata_dict, key_name=None, folder='', key=None, bucket_name=''):
        """
        In order to update metadata on an object, need to copy (==resave) it to same location
        Note: pass in EITHER an actual key or else a key name (for lookup)
        """
        bucket = self._get_new_bucket_or_default(bucket_name)
        key = key or bucket_name.lookup(self._full_key_name(folder, key_name))
        key.metadata.update(metadata_dict)
        # MJM - note, removing preserve ACL since we should not need it here
        key.copy(bucket.name, key.name, key.metadata)

    def copy_and_return_errors(self, key_name_list, from_folder='', from_bucket_name='',
                               to_folder='', to_bucket_name='', storage_class='REDUCED_REDUNDANCY'):
        """
        Take a list of key names and moves them from the specified bucket and folder to the specified bucket and folder.
        If no folder(s) is specified, the top level of the bucket is used.
        If no bucket(s) is specified, the default bucket is used
        Returns a list of all errors encountered
        """

        assert storage_class in ['STANDARD', 'REDUCED_REDUNDANCY']

        from_bucket = self._get_new_bucket_or_default(from_bucket_name)
        to_bucket = bucket = self._get_new_bucket_or_default(to_bucket_name)

        to_and_from_key_names = [
            (self._full_key_name(to_folder, key_name),
             self._full_key_name(from_folder, key_name))
            for key_name in key_name_list
        ]

        errors = []

        for to_from_names in to_and_from_key_names:
            try:
                to_bucket.copy_key(
                    new_key_name=to_from_names[0],
                    src_bucket_name=from_bucket.name,
                    src_key_name=to_from_names[1],
                    storage_class=storage_class
                )
            except S3ResponseError as e:
                errors.append(e)

        return errors

    def _remove_folder_from_name(self, key, folder):
        key.name = key.name.replace(folder, '')
        return key

    def _get_s3_key_for_dump(self, key_name, folder, bucket):
        """
        Takes care of the process of getting an s3 key object set up for saving, within the correct bucket and
        with a name that takes the folder (if any) into account.
        """
        s3_key = Key(bucket)
        s3_key.key = self._full_key_name(folder, key_name)
        return s3_key

    def _get_s3_key_for_load(self, key_name, folder, bucket):
        """
        Takes care of the process of getting an s3 key object set up for loading, within the correct bucket and
        with a name that takes the folder (if any) into account.
        """
        return bucket.lookup(self._full_key_name(folder, key_name))

    def _full_key_name(self, folder, key_name):
        folder_path = self.base_folder + self._ensure_good_folder_name(folder)
        return folder_path + key_name

    def _ensure_good_folder_name(self, folder_name):

        # we are going to append this to key name, so ensure it is not null
        folder_name = folder_name or ''

        if folder_name:
            if not folder_name.endswith('/'):
                folder_name += '/'
        return folder_name

    def _get_new_bucket_or_default(self, bucket_name):
        if bucket_name:
            return self.connection.get_bucket(bucket_name)
        else:
            return self.bucket