__author__ = 'thorwhalen'

from boto.s3.connection import Location
from boto.s3.key import Key
import boto.s3
import tempfile
import pickle
import os
import sys
from ut.util.importing import get_environment_variable

S3_DEFAULT_REGION = 'eu-west-1'


class S3:
    """
    Storage to Amazon S3.
    """

    def __init__(self, bucket_name, region=S3_DEFAULT_REGION):
        """
        Creates an instance with a handle on the S3 bucket corresponding to bucket_name. If bucket_name does not
        exist, it is created.

        :type bucket_name: str
        """
        self.conn = boto.s3.connect_to_region(
            region,
            aws_access_key_id=get_environment_variable('VEN_S3_SECRET'),
            aws_secret_access_key=get_environment_variable('VEN_S3_ACCESS_KEY'),
        )
        if bucket_name not in [b.name for b in self.conn.get_all_buckets()]:
            self.conn.create_bucket(bucket_name, location=Location.EU)
        self.bucket = self.conn.get_bucket(bucket_name)

    def save_file(self, key_name, file, folder=''):
        """
        Saves the key, file pair to S3 bucket and prints progress as dots

        :param key: the key to save
        :param value: the file to save
        """

        def percent_cb(complete, total):
            sys.stdout.write('.')
            sys.stdout.flush()

        k = Key(self.bucket)
        k.key = folder + key_name
        k.set_contents_from_file(file, cb=percent_cb, num_cb=10)

    def save_object(self, key_name, value, folder=''):
        """
        Saves the key, value pair to S3 bucket via file serialization

        :param key: the key to save
        :param value: the value to save
        """
        k = Key(self.bucket)
        k.key = folder + key_name
        with tempfile.NamedTemporaryFile() as tempf:
            tempf = tempfile.NamedTemporaryFile()
            pickle.dump(value, tempf)
            tempf.seek(0)
            k.set_contents_from_file(tempf)

    def read_object(self, key_name, folder=''):
        """
        Reads the value corresponding to the key from S3 bucket via a file, then loads the object into memory and
        returns it.

        :param key: key to read
        :return: value
        """
        with tempfile.NamedTemporaryFile() as tempf:
            k = self.bucket.lookup(folder + key_name)
            k.get_contents_to_file(tempf)
            tempf.seek(0)
            val = pickle.load(tempf)
            return val

    def save_string(self, key_name, value, folder=''):
        """
        Saves the key and value, with the assumption that the value is a valid string.

        :param key: the key
        :param value: the value
        """
        k = Key(self.bucket)
        k.key = folder + key_name
        k.set_contents_from_string(value)

    def read_string(self, key_name, folder=''):
        """
        Reads the value corresponding to the key from S3, assuming the value is a string.

        :param key: the key
        :return: the value (as string)
        """
        k = self.bucket.lookup(folder + key_name)
        return k.get_contents_to_string()

    def download(
        self, key_name, filepath='', s3_folder='', local_folder='', suffix='', prefix=''
    ):
        if not filepath:
            filepath = key_name
        filepath = local_folder + prefix + filepath + suffix
        print(filepath)
        if not os.path.exists(filepath):
            k = self.bucket.lookup(s3_folder + key_name)
            k.get_contents_to_filename(filepath)
