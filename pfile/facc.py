

import os
from glob import glob
from shutil import copyfile
import re

def ensure_slash_suffix(str):
    if str:
        if str[-1] != '/':
            return str + '/'
    return str


class LocalFileAccess(object):

    def isdir(self, path):
        return os.path.isdir(path)

    def isfile(self, path):
        return os.path.isfile(path)

    def mkdir(self, path):
        os.mkdir(path)

    def join(self, p1, p2):
        return os.path.join(p1, p2)

    def listdir(self, path):
        return [os.path.join(path, l) for l in os.listdir(path)]

    def glob(self, pattern):
        return glob(pattern)

    def open(self, path, mode='rb'):
        return open(path, mode)

    def getsize(self, path):
        return os.path.getsize(path)

    def basename(self, path):
        return os.path.basename(path)

    def copyfile(self, src, dst):
        copyfile(src, dst)

    def rm(self, path):
        os.remove(path)

    def rmdir(self, path):
        os.rmdir(path)

DFLT_FILE_PATTERN = '.*'

class FilesAccess(object):
    def __init__(self, root, file_pattern=DFLT_FILE_PATTERN):
        """
        :param root: Folder where the data is stored
        """

        # TODO: Prefered way to make it work (but need to make it work):
        # root, _s3_access, _s3_secret = extract_s3_creds_from_root(root)
        # self.s3_access = _s3_access or s3_access
        # self.s3_secret = _s3_secret or s3_secret

        self.root = ensure_slash_suffix(os.path.expanduser(root))
        self.fs = LocalFileAccess()

        self.file_pattern = re.compile(file_pattern)

    # @classmethod
    # def for_local(cls, root, file_pattern=DFLT_FILE_PATTERN):


    def filepath_list(self, file_pattern=None):
        if file_pattern is None:
            return list(filter(self.file_pattern.match, self.fs.listdir(self.root)))
        else:
            file_pattern = re.compile(file_pattern)
            return list(filter(file_pattern.match, self.fs.listdir(self.root)))

    # TODO: if the import script is run on a directory that has 'can' subdirectory, causes problems
    def folderpath(self, foldername):
        if self.fs.isdir(foldername):
            return ensure_slash_suffix(foldername)
        else:
            folder = ensure_slash_suffix(self.fs.join(self.root, foldername))
            if not self.fs.isdir(folder):
                self.fs.mkdir(folder)
            return folder

    def filepath(self, filename):
        if self.fs.isfile(filename):
            return filename
        else:
            return self.fs.join(self.root, filename)

    def filepath_search(self, filepath, search_paths=()):
        """
        Searches for a filepath that actually resolves to a file (i.e. self.fs.isfile(filepath) returns True.
        Will first check the input filepath as is.
        If not found, will then loop through search_paths in that order, and try to find the file with the listed paths
        (i.e. prefixes).
        If it exhausted all possibilities, will return None.
        :param filepath: The filepath (seed) to search for
        :param search_paths: A list of path prefixes to look for the file
        :return: A filepath that will be found (if using open(filepath) for example), or None if not found
        """
        if self.fs.isfile(filepath):
            return filepath
        else:
            for path in search_paths:
                candidate_filepath = self.fs.join(path, filepath)
                if self.fs.isfile(candidate_filepath):
                    return candidate_filepath
            return None  # if couldn't find a match at this point, return None
