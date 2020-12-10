from contextlib import suppress

with suppress(ModuleNotFoundError):
    from lag import *

import os
import contextlib


def clog(*args, condition=True, log_func=print, **kwargs):
    if condition:
        return log_func(*args, **kwargs)


@contextlib.contextmanager
def cd(newdir, verbose=True):
    """Change your working directory, do stuff, and change back to the original"""
    _clog = partial(clog, condition=verbose, log_func=print)
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        _clog(f'cd {newdir}')
        yield
    finally:
        _clog(f'cd {prevdir}')
        os.chdir(prevdir)
    # from pathlib import Path
    # _clog("Called before cd", Path().absolute())
    # with cd(Path.home()):
    #     if verbose: print("Called under cd", Path().absolute())
    # _clog("Called after cd and same as before", Path().absolute())
