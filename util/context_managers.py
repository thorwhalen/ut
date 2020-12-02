from warnings import warn
from ut import ModuleNotFoundIgnore

with ModuleNotFoundIgnore():
    from lag import *

import os
import contextlib


@contextlib.contextmanager
def cd(newdir, verbose=True):
    """Change your working directory, run the function, and change back to the original"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        if verbose: print(f'cd {newdir}')
        yield
    finally:
        if verbose: print(f'cd {prevdir}')
        os.chdir(prevdir)
    from pathlib import Path
    if verbose: print("Called before cd", Path().absolute())
    with cd(Path.home()):
        if verbose: print("Called under cd", Path().absolute())
    if verbose: print("Called after cd and same as before", Path().absolute())
