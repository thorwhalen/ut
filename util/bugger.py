"""Debugging tools"""
__author__ = 'thor'

import ut.pfile.accessor as pfile_accessor
import os
from ut.util.importing import get_environment_variable
import pdb

PROJECT_PATH = get_environment_variable('PY_PROJ_FOLDER')

#import os
#PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# THOR_KHAN_CODE_PATH = "/D/Dropbox/dev/py/proj/khan/"
# MATT_KHAN_CODE_PATH = "/Users/mattjmorris/Dev/python/khan/"

pdbrc_file = '.pdbrc'

# constant .pdbrc aliases

# define pdbrc_prefix
pdbrc_prefix = "##### PDBRC PREFIX ###########################" + "\n"
pdbrc_prefix = pdbrc_prefix + "\n" + "# Print a dictionary, sorted. %1 is the dict, %2 is the prefix for the names."
pdbrc_prefix = pdbrc_prefix + "\n" + 'alias p_ for k in sorted(%1.keys()): print "%s%-15s= %-80.80s" % ("%2",k,repr(%1[k]))'
pdbrc_prefix = pdbrc_prefix + "\n" + ""
pdbrc_prefix = pdbrc_prefix + "\n" + "# Print the instance variables of a thing."
pdbrc_prefix = pdbrc_prefix + "\n" + "alias pi p_ %1.__dict__ %1."
pdbrc_prefix = pdbrc_prefix + "\n" + ""
pdbrc_prefix = pdbrc_prefix + "\n" + "# Print the instance variables of self."
pdbrc_prefix = pdbrc_prefix + "\n" + "alias ps pi self"
pdbrc_prefix = pdbrc_prefix + "\n" + ""
pdbrc_prefix = pdbrc_prefix + "\n" + "# Print the locals."
pdbrc_prefix = pdbrc_prefix + "\n" + "alias pl p_ locals() local:"
pdbrc_prefix = pdbrc_prefix + "\n" + ""
pdbrc_prefix = pdbrc_prefix + "\n" + "# Next and list, and step and list."
pdbrc_prefix = pdbrc_prefix + "\n" + "alias nl n;;l"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias sl s;;l"
pdbrc_prefix = pdbrc_prefix + "\n" + ""
pdbrc_prefix = pdbrc_prefix + "\n" + "# Short cuts for walking up and down the stack"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias uu u;;u"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias uuu u;;u;;u"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias uuuu u;;u;;u;;u"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias uuuuu u;;u;;u;;u;;u"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias dd d;;d"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias ddd d;;d;;d"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias dddd d;;d;;d;;d"
pdbrc_prefix = pdbrc_prefix + "\n" + "alias ddddd d;;d;;d;;d;;d"
pdbrc_prefix = pdbrc_prefix + "\n" + ""
# pdbrc_prefix = pdbrc_prefix + "\n" + "from datetime import datetime"
# pdbrc_prefix = pdbrc_prefix + "\n" + "alias nn start_time=datetime.now();; n;; l;; print 'elapsed: %.03f'%(datetime.now() - start_time).total_seconds();;"
pdbrc_prefix = pdbrc_prefix + "\n" + ""
pdbrc_prefix = pdbrc_prefix + "\n" +  '#Print instance variables (usage "pi classInst")'
pdbrc_prefix = pdbrc_prefix + "\n" +  'alias pi for k in %1.__dict__.keys(): print "%1.",k,"=",%1.__dict__[k]'
pdbrc_prefix = pdbrc_prefix + "\n" +  '#Print instance variables in self'
pdbrc_prefix = pdbrc_prefix + "\n" +  'alias ps pi self'
pdbrc_prefix = pdbrc_prefix + "\n\n"
pdbrc_prefix = pdbrc_prefix + "##### ADDITIONAL LINES ###########################" + "\n"


def init():
    with open(pdbrc_file, 'w') as f:
        f.write(pdbrc_prefix)

def add(s):
    if not isinstance(s, str):
        s = "\n".join(s)
    with open(pdbrc_file, 'a') as f:
        f.write(s)
        f.write('\n')

def init_and_add(s):
    init()
    add(s)

def print_pdbrc():
    with open(pdbrc_file, 'r') as f:
        print(f.read())

# def break_and_go(command, filename, lineno=1):
#     init()
#     add_breakpoint(filename, lineno)
#     add("continue")
def add_breakpoint(filename, lineno=1):
    with open(pdbrc_file, 'a') as f:
        add("break %s:%d" % (filename, lineno))


class Bugger(object):
    def __init__(self, relative_root='', root_path=PROJECT_PATH):
        self.facc = pfile_accessor.for_local(root_folder=root_path, relative_root=relative_root, extension='.py', force_extension=True)
        print("Bugger created for path: %s" % self.facc.root_folder())
        if not os.path.exists(self.facc.root_folder()):
            print("!!! This path doesn't exist: May want to verify it")

    def init(self):
        init()

    def add_breakpoint(self, filename, lineno=1):
        add_breakpoint(filename=self.facc(filename), lineno=lineno)

    def init_add_breakpoint(self, filename, lineno=1):
        init()
        add_breakpoint(filename=self.facc(filename), lineno=lineno)

    def add(self, s):
        add(s)

    def init_and_add(self, s):
        init_and_add(s)

    def print_pdbrc(self):
        with open(pdbrc_file, 'r') as f:
            print(f.read())

    def run(self, s):
        pdb.run(s)

