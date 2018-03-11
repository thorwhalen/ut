from __future__ import division

from pkg_resources import get_distribution, DistributionNotFound, RequirementParseError


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
