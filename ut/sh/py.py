import sys


def add_to_pythonpath_if_not_there(paths):
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        if p not in sys.path:
            sys.path.append(p)
