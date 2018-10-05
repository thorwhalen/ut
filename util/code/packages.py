from __future__ import division

import os
import re
from pkg_resources import get_distribution, DistributionNotFound, RequirementParseError
from ut.pfile.iter import get_filepath_iterator
import ut.pfile.to


def read_requirements(requirements_file):
    with open(requirements_file, 'r') as f:
        return f.read().splitlines()


def get_module_name(package, on_error='raise'):
    try:
        t = list(get_distribution(package)._get_metadata('top_level.txt'))
        if t:
            return t[0]
        else:
            return None
    except Exception as e:
        if on_error == 'raise':
            raise
        elif on_error == 'error_class':
            return e.__class__.__name__
        else:
            return on_error  # just the value specified by on_error


def set_of_to_level_import_names(folder_path_to_scan=None, filt_set=None):
    """
    Scan a folder recursively, picking up the .py files, seeing what names are imported
    (i.e. from NAME.* and import NAME.*), collecting the full set of these, and then
    intersecting with the filt_set.
    :param folder_path_to_scan: Folder path to scan recursively
    :param filt_set: names to filter in (i.e. match from the full set of import names found
    :return: A set of import names found (of those in the filt_set that is)
    """
    if folder_path_to_scan is None:
        folder_path_to_scan = '/D/Dropbox/dev/py/proj/span/'
    if filt_set is None:
        filt_set = set(os.listdir('/D/Dropbox/dev/py/proj/')).union('oto', 'scan')

    p = re.compile('import \w+|from \w+')
    cumul = set()
    for f in get_filepath_iterator(folder_path_to_scan, '.py$'):
        s = ut.pfile.to.string(f)
        x = map(lambda x: x.split(' ')[1], p.findall(s))
        cumul = cumul.union(x)
    return cumul.intersection(filt_set)
