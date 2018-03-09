from __future__ import division

from numpy import unique
from collections import Counter
import re
import inspect
import os

from ut.util.code.packages import get_package_name, read_requirements

from ut.pfile.iter import get_filepath_iterator


def mk_single_package_import_regex(package_name):
    return re.compile("(?<=from) {package_name}|(?<=import) {package_name}".format(package_name=package_name))


def mk_multiple_package_import_regex(package_names):
    if isinstance(package_names, basestring):
        package_names = [package_names]
    return re.compile('|'.join(map(lambda x: mk_single_package_import_regex(x).pattern, package_names)))


def packages_in_module(module, package_names):
    if isinstance(package_names, (tuple, list)):
        package_names = mk_multiple_package_import_regex(package_names)
    if inspect.ismodule(module):
        module = inspect.getsourcefile(module)
    if module.endswith('__init__.py'):
        module = os.path.dirname(module)
    if os.path.isdir(module):
        c = Counter()
        it = get_filepath_iterator(module, pattern='.py$')
        it.next()
        for _module in it:
            c.update(packages_in_module(_module, package_names))
        return c
    elif not os.path.isfile(module):
        raise ValueError("module file not found: {}".format(module))

    with open(module) as fp:
        module_contents = fp.read()
    return Counter(map(lambda x: x[1:], unique(package_names.findall(module_contents))))


def requirements_packages_in_module(module, requirements):
    if isinstance(requirements, basestring) and os.path.isfile(requirements):
        with open(requirements) as fp:
            requirements = fp.read().splitlines()

    p = re.compile('^[^=]+')
    packages = list()
    for x in requirements:
        try:
            xx = p.findall(x)
            if xx:
                package_name = get_package_name(xx[0])
                packages.append(package_name)
        except Exception as e:
            print("Error with {}\n  {}".format(x, e))

    return packages_in_module(module, packages)
