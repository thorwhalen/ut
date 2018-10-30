from __future__ import division

import os
import re

key_file_re = re.compile('setup.py')


def dir_has_a_setup_py(dirpath):
    return any(filter(key_file_re.match, os.listdir(dirpath)))


def find_folders_with_setup_py(rootdir):
    """
    Returns a list of folder paths that are under rootdir and contain a setup.py
    :param rootdir: directory to search for (subdirectories)
    :return: a list of folder paths
    """
    rootdir = os.path.abspath(rootdir)
    cumul = list()
    for f in filter(lambda x: not x.startswith('.'), os.listdir(rootdir)):
        filepath = os.path.join(rootdir, f)
        if os.path.isdir(filepath):
            if dir_has_a_setup_py(filepath):
                cumul.append(filepath)
    return cumul


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--rootdir",
                        help="directory to search for (subdirectories)")
    parser.add_argument("--display",
                        help="How to display the list: print or for_pythonpath", default='print')


    args = parser.parse_args()
    args = dict(vars(args))

    rootdir = args['rootdir']
    display = args.get('display', 'print')

    cumul = find_folders_with_setup_py(rootdir)

    if display == 'print':
        for f in cumul:
            print(f)
    elif display == 'for_pythonpath':
        print(":".join(cumul))
    else:
        print("Don't know how to display")
