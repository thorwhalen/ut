__author__ = 'thorwhalen'

from serialize.s3 import S3
import os

class MS_S3(S3):
    def download(self, key_name, filepath, s3_folder='', local_folder='', suffix='', prefix=''):
        filepath = os.path.join(local_folder,prefix + filepath + suffix)
        if not os.path.exists(filepath):
            k = self.bucket.lookup(os.path.join(s3_folder,key_name))
            k.get_contents_to_filename(filepath)
