import sys


def add_to_pythonpath_if_not_there(paths):
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        if p not in sys.path:
            sys.path.append(p)


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([add_to_pythonpath_if_not_there])
    parser.dispatch()
