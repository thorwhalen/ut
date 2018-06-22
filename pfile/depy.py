from __future__ import division

import shutil
import os
import re
import ut.util.time as utime
import py_compile
from itertools import imap
from glob import iglob
from datetime import datetime as dt

file_sep = os.path.sep
second_ms = 1000.0
epoch = dt.utcfromtimestamp(0)


def utcnow_timestamp():
    return (dt.utcnow() - epoch).total_seconds()


def utcnow_ms():
    return (dt.utcnow() - epoch).total_seconds() * second_ms


def iter_relative_files_and_folder(root_folder):
    if root_folder[-1] != file_sep:
        root_folder += file_sep
    return imap(lambda x: x.replace(root_folder, ''), iglob(root_folder + '*'))


def pattern_filter(pattern):
    pattern = re.compile(pattern)

    def _pattern_filter(s):
        return pattern.search(s) is not None

    return _pattern_filter


def recursive_file_walk_iterator_with_name_filter(root_folder, filt='', return_full_path=True):
    if isinstance(filt, basestring):
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


def ensure_slash_suffix(x):
    if not x.endswith(os.path.sep):
        return x + os.path.sep
    else:
        return x


def pyc_of_py(py_filepath):
    basename, ext = os.path.splitext(py_filepath)
    if ext == '.py':
        return basename + '.pyc'
    else:
        raise ValueError("Wasn't a .py file: {}".format(py_filepath))


def py_of_pyc(pyc_filepath):
    basename, ext = os.path.splitext(pyc_filepath)
    if ext == '.pyc':
        return basename + '.py'
    else:
        raise ValueError("Wasn't a .pyc file: {}".format(pyc_filepath))


def pyc_of_py_exists(py_filepath):
    pyc_filepath = pyc_of_py(py_filepath)
    return os.path.isfile(pyc_filepath)


def parent_folder(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def depy(folderpath, backup_folderpath=None):
    folderpath = ensure_slash_suffix(folderpath)
    if backup_folderpath is None:
        parent = parent_folder(folderpath)
        name = os.path.basename(folderpath[:-1])
        backup_folderpath = os.path.join(parent, "depy_{}_bak_{}".format(name, int(utime.utcnow_ms())))

    if backup_folderpath:
        shutil.copytree(folderpath, backup_folderpath)

    not_removed_py_filepaths = list()
    n_removed = 0
    for py_filepath in recursive_file_walk_iterator_with_name_filter(folderpath, filt='.py$', return_full_path=True):

        pyc_filepath = pyc_of_py(py_filepath)
        py_compile.compile(py_filepath, pyc_filepath)  # recompute the pyc systematically
        if os.path.isfile(pyc_filepath):
            os.remove(py_filepath)
            n_removed += 1
        else:
            not_removed_py_filepaths.append(py_filepath)
    print("\n{} .py files were successfully removed.\n".format(n_removed))

    if len(not_removed_py_filepaths) > 0:
        print("{} .py files were not removed because no pyc could be made for it "
              "(the .py probably doesn't work anyway)".format(len(not_removed_py_filepaths)))
        print("Here's the list of those not removed files:")
        for f in not_removed_py_filepaths:
            print("    {}".format(f))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    args_def = {
        'folderpath': {
            'help': 'The folder to remove all .py files from (if they have a .pyc, or one can be generated)',
            'type': str
        },
        'backup_folderpath': {
            'help': 'The folder name one should use to backup the whole folderpath before removing .py files. '
                    'Default will be chosen and timestamped. If you do not wish for a backup, you should enter "False"',
            'default': None,
            'type': str
        }
    }
    for arg_name, params in args_def.iteritems():
        parser.add_argument("--" + arg_name, **params)

    p, other_kwargs = parser.parse_known_args()

    # p = parser.parse_args()
    kwargs = vars(p)
    if kwargs['backup_folderpath'] == 'False':
        kwargs['backup_folderpath'] = False

    depy(**kwargs)
