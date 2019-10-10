# zipmod.py - make a zip archive consisting of Python modules and their dependencies as reported by modulefinder
# To use: cd to the directory containing your Python module tree and type
# $ python zipmod.py archive.zip mod1.py mod2.py ...
# Only modules in the current working directory and its subdirectories will be included.
# Written and tested on Mac OS X, but it should work on other platforms with minimal modifications.

import modulefinder
import os
import sys
import zipfile


def main(output, *mnames):
    mf = modulefinder.ModuleFinder()
    for mname in mnames:
        mf.run_script(mname)
    cwd = os.getcwd()
    zf = zipfile.ZipFile(output, 'w')
    for mod in mf.modules.values():
        if not mod.__file__:
            continue
        modfile = os.path.abspath(mod.__file__)
        if os.path.commonprefix([cwd, modfile]) == cwd:
            zf.write(modfile, os.path.relpath(modfile))
    zf.close()

if __name__ == '__main__':
    main(*sys.argv[1:])