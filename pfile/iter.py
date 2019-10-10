

import os
import re

from glob import iglob

__author__ = 'thor'


file_sep = os.sep


class FilePatterns(object):
    NOT_HIDDEN_FILE = '^[^.].+'
    WAV_EXTENSION = '.wav$'


def get_file_iterator(root_folder,
                      filt,
                      return_full_path=True,
                      apply_pattern_to_full_path=False):
    pass


def get_filepath_iterator(root_folder,
                          pattern='',
                          return_full_path=True,
                          apply_pattern_to_full_path=False):
    if apply_pattern_to_full_path:
        return recursive_file_walk_iterator_with_name_filter(root_folder, pattern, return_full_path)
    else:
        return recursive_file_walk_iterator_with_filepath_filter(root_folder, pattern, return_full_path)


def iter_relative_files_and_folder(root_folder):
    if not root_folder.endswith(file_sep):
        root_folder += file_sep
    return map(lambda x: x.replace(root_folder, ''), iglob(root_folder + '*'))


def pattern_filter(pattern):
    pattern = re.compile(pattern)

    def _pattern_filter(s):
        return pattern.search(s) is not None

    return _pattern_filter


def recursive_file_walk_iterator_with_name_filter(root_folder, filt='', return_full_path=True):
    if isinstance(filt, str):
        filt = pattern_filter(filt)
    # if isinstance(pattern, basestring):
    #     pattern = re.compile(pattern)
    for name in iter_relative_files_and_folder(root_folder):
        full_path = os.path.join(root_folder, name)
        if os.path.isdir(full_path):
            for entry in recursive_file_walk_iterator_with_name_filter(full_path, filt, return_full_path):
                yield entry
        else:
            if os.path.isfile(full_path):
                if filt(name):
                    if return_full_path:
                        yield full_path
                    else:
                        yield name


def recursive_file_walk_iterator_with_filepath_filter(root_folder, filt='', return_full_path=True):
    if isinstance(filt, str):
        filt = pattern_filter(filt)
    for name in iter_relative_files_and_folder(root_folder):
        full_path = os.path.join(root_folder, name)
        if os.path.isdir(full_path):
            for entry in recursive_file_walk_iterator_with_filepath_filter(full_path, filt, return_full_path):
                yield entry
        else:
            if os.path.isfile(full_path):
                if filt(full_path):
                    if return_full_path:
                        yield full_path
                    else:
                        yield name
