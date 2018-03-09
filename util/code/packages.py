from __future__ import division

import pkg_resources


def read_requirements(requirements_file):
    with open(requirements_file, 'r') as f:
        return f.read().splitlines()


def get_package_name(package):
    t = list(pkg_resources.get_distribution(package)._get_metadata('top_level.txt'))
    if t:
        return t[0]
    else:
        return None


