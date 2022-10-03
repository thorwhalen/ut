"""Utils to work with pip resources"""
__author__ = 'thorwhalen'

import re
import os
import sys
import subprocess
import inspect
from collections import Counter
from ut.pfile.to import string as file_to_str
from ut.pfile.iter import get_filepath_iterator
import imp


def extract_package_list_from_filepath(filepath):
    s = file_to_str(filepath)
    ss = s.split('\n')
    import_capture_p = re.compile('(import|from)\s([^\.\s\n]+)')
    return [x.group(2) for x in [_f for _f in map(import_capture_p.match, ss) if _f]]


def count_packages_of_all_py_files_recursively_under(root_folder):
    c = Counter()
    for filepath in get_filepath_iterator(root_folder, pattern='.py$'):
        c.update(extract_package_list_from_filepath(filepath))
    return c


PIPFREEZE_LIST_FOLDER = '/D/Dropbox/dev/py/info/'
PIPFREEZE_AIR = '/D/Dropbox/dev/py/info/pipfreeze_air_sem.txt'
PIPFREEZE_PRO = '/D/Dropbox/dev/py/info/pipfreeze_pro_sem.txt'
PIPFREEZE_MONK = '/D/Dropbox/dev/py/info/pipfreeze_monk.txt'
PIPFREEZE_ALL = [PIPFREEZE_AIR, PIPFREEZE_PRO]


def put_pip_freeze_output_into_file(output_file_path, environment=None):
    if environment:
        os.system('workon '.format(environment))
    s = os.system('pip freeze > {}'.format(output_file_path))
    print(s)


def import_subdirs_of(dir):
    subdirs = [x for x in get_immediate_subdirectories(dir) if x[0] != '.']
    for s in subdirs:
        print('importing to sys path: {}'.format(s))
        sys.path.append(s)


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


# needs to be primed with an empty set for loaded (THIS IS SOMEONE ELSES CODE. NOT SURE IF IT WORKS
def recursively_reload_all_submodules(module, loaded=None):
    for name in dir(module):
        member = getattr(module, name)
        if inspect.ismodule(member) and member not in loaded:
            recursively_reload_all_submodules(member, loaded)
    loaded.add(module)
    imp.reload(module)


def mk_list_file_name(computer_name, environment=None):
    return 'pipfreeze_{}_{}.txt'.format(computer_name, environment)


def mk_pip_freeze_list_file(computer_name, environment=None):
    if environment:
        os.system('workon '.format(environment))
    s = os.system(
        'pip freeze > {}'.format(
            PIPFREEZE_LIST_FOLDER
            + mk_list_file_name(computer_name, environment=environment)
        )
    )
    print(s)


def pipfreeze_list(filepath, unversioned=True):
    if filepath == 'all':
        package_list = []
        for f in PIPFREEZE_ALL:
            package_list = list(
                set(package_list + pipfreeze_list(f, unversioned=unversioned))
            )
    else:
        filepath = get_filepath_of_pipinstall_list(filepath)
        with open(filepath, 'r') as myfile:
            package_list = [x.replace('\n', '') for x in myfile.readlines()]
    if unversioned == True:
        package_list = [re.match('[^=]*', x).group() for x in package_list]
    return package_list


def get_filepath_of_pipinstall_list(spec):
    if spec == 'air':
        return PIPFREEZE_AIR
    elif spec == 'pro':
        return PIPFREEZE_PRO
    elif spec == 'monk':
        return PIPFREEZE_MONK
    else:
        return spec


def missing_pip_installs(compare_set, unversioned=True):
    if isinstance(compare_set, str):
        if compare_set == 'air':
            compare_set = pipfreeze_list(PIPFREEZE_AIR, unversioned=unversioned)
        elif compare_set == 'pro':
            compare_set = pipfreeze_list(PIPFREEZE_PRO, unversioned=unversioned)
        else:
            compare_set = pipfreeze_list(compare_set, unversioned=unversioned)
    else:
        assert isinstance(compare_set, list)
    all_set = pipfreeze_list('all', unversioned=unversioned)
    return list(set(all_set) - set(compare_set))


# def pipfreeze_air_minus_pro(unversioned=True):
#     if unversioned==True:
#         air = unversioned_pipfreeze_list(PIPFREEZE_AIR)
#         pro = unversioned_pipfreeze_list(PIPFREEZE_PRO)
#     else:
#         air = lines_to_list(PIPFREEZE_AIR)
#         pro = lines_to_list(PIPFREEZE_PRO)
#     return list(set(air)- set(pro))
#
# def pipfreeze_pro_minus_air(unversioned=True):
#     if unversioned==True:
#         air = unversioned_pipfreeze_list(PIPFREEZE_AIR)
#         pro = unversioned_pipfreeze_list(PIPFREEZE_PRO)
#     else:
#         air = lines_to_list(PIPFREEZE_AIR)
#         pro = lines_to_list(PIPFREEZE_PRO)
#     return list(set(pro)- set(air))


def pip_install_list(package_names, environment='sem'):
    if environment:
        os.system('workon '.format(environment))
    for pak in package_names:
        # TODO: Not sure this actually installs it in the appropriate environment...
        print('------- pip install {}'.format(pak))
        s = subprocess.check_output('pip install {}'.format(pak).split(' '))
        print(s)


def pip_install_missing(compare_set, environment='sem'):
    missing_packages = missing_pip_installs(compare_set, unversioned=True)
    pip_install_list(missing_packages, environment=environment)
