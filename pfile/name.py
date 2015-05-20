__author__ = 'thorwhalen'

import os
import glob
from ut.pcoll.op import ismember
import re


def recursive_file_walk_iterator(directory, pattern=''):
    if isinstance(pattern, basestring):
        pattern = re.compile(pattern)
    # return pattern
    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        # print full_path
        if os.path.isdir(full_path):
            for entry in recursive_file_walk_iterator(full_path, pattern):
                yield entry
        elif os.path.isfile(full_path):
            # print full_path
            if pattern.search(full_path):
                yield full_path


def add_extension_if_not_present(filename, ext=None):
    if get_extension(filename)=='' and ext:
        return replace_extension(filename, ext)
    return filename


def get_extension(filename):
    (root, ext) = os.path.splitext(filename)
    return ext


def files_of(folder):
    """
    returns a list of files in the folder, EXCLUDING hidden files
    """
    return glob.glob(os.path.join(folder, '*'))


def files_of_folder(folder):
    f = []
    for (dirpath, dirname, filenames) in os.walk(folder):
        f.extend(filenames)
        break
    return f


def replace_folder_and_ext(filename, newfolder, ext):
    return replace_extension(os.path.join(newfolder,os.path.basename(filename)),ext)


def replace_folder(filename, newfolder):
    return os.path.join(newfolder,os.path.basename(filename))


def replace_extension(filename, ext):
    """
    replaces the extension of a filename by the input extension ext
    """
    if ext=='':
        return os.path.splitext(filename)[0]
    else:
        return os.path.splitext(filename)[0] + ensure_dot_prefix(ext)


def ensure_parent_folder(filename, default_parent_folder):
    """
    returns the filename but with a parent folder prefixed if it had no parent folder (i.e. if there were no /)
    """
    if os.path.split(filename)[0]:
        return os.path.join(default_parent_folder, filename)
    else:
        return filename

def ensure_dot_prefix(ext):
    if ext:
        if ext[0] != '.':
            ext = '.' + ext
    return ext


def ensure_slash_suffix(str):
    if str:
        if str[-1]!='/':
            str = str + '/'
    return str


def fileparts(file):
    """

    :param file: a filepath
    :return: the root, name, and extension of the pfile
    """
    import os.path
    (root, ext) = os.path.splitext(file)
    (x, name) = os.path.split(root)
    if root==name:
        return ('', name, ext)
    else:
        return (root, name, ext)


def get_highest_level_folder(filepath):
    file_parts = filepath.split('/')
    if not file_parts[0]:
        # if file_parts[0] is empty
        if len(file_parts) >= 2: # and there is some more...
            return file_parts[1]
        else:
            return file_parts[0]
    else:
        return file_parts[0]


def fullfile(root, name, ext):
    """

    :param root:
    :param name:
    :param ext:
    :return: the root, name, and extension of the pfile
    """
    import os.path
    return os.path.join(root,name+ext)


# input: filename
# output: True if the extension (.csv, .tab, or .txt) looks like it might be a delim pfile
def is_delim_file(dataname):
    '''
    input: filename
    output: True if the extension (.csv, .tab, or .txt) looks like it might be a delim pfile
    '''
    root, name, ext = fileparts(dataname)
    if ext and ismember(ext,['.csv','.tab','.txt']):
        return True
    else:
        return False

# input: dataname
# output: csv pfile path for this dataname, if such a pfile exists, looking for files that have the template
#   data_folder + dataname + csvExtensions
# given the list of csvExtensions (default ['.csv', '.tab', '.txt']) and list of data_folders
# where the list of csvExtensions is traversed in priority, and only if dataname has no extension already
# and the list of data_folders is traversed in second priority, and only if dataname is a simple filename (i.e. has no head path)
#
# Note: If no existing pfile is found, the function returns an empty string
#
#   if dataname includes an extension which is in csvExtensions
#     dataname
#       and if dataname has no heading path...
#         'csv/' + dataname
#         data_folder + dataname
#         data_folder + '/csv/' + dataname
#   if dataname has no extension
#     dataname + csvExtensions
#       and if dataname has no heading path...
#         'csv/' + dataname + csvExtensions
#         data_folder + dataname + csvExtensions
#         data_folder + '/csv/' + dataname + csvExtensions
def delim_file(dataname, data_folder=['','csv'], csvExtensions=['.csv', '.tab', '.txt']):
    import os.path
    # set up lists of folders and extensions we'll be looking through
    root,name,ext = fileparts(dataname)
    if ext and ismember(ext,csvExtensions): # if dataname had a permissable extension
        csvExtensions = [ext]
    if root: # if dataname has a path header (i.e. is specified by a full path)
        data_folder = [root]
    else:
        if isinstance(data_folder,list):
            tail_options = data_folder
            data_folder = ['']
        else:
            tail_options = ['','data','daf']
            data_folder = [data_folder]
        data_folder = [os.path.join(f,t) for f in data_folder for t in tail_options]
        # look through possibilities until a pfile is found (or not)
    for folder in data_folder:
        for ext in csvExtensions:
            try_filename = fullfile(folder,name,ext)
            if os.path.exists(try_filename):
                return try_filename
    return '' # if no pfile was found

# input: dataname
# output: pfile path for this dataname, if such a pfile exists, looking for files that have the template
#   data_folder + dataname + fileExtensions
# NOTE: Same as delim_file (see this function for more details), but with different fileExtensions defaults
def data_file(dataname, data_folder=['','data','daf'], fileExtensions=['']):
    import os.path
    # set up lists of folders and extensions we'll be looking through
    root,name,ext = fileparts(dataname)
    if ext and ismember(ext,fileExtensions): # if dataname had a permissable extension
        fileExtensions = [ext]
    if root: # if dataname has a path header (i.e. is specified by a full path)
        data_folder = [root]
    else:
        if isinstance(data_folder,list):
            tail_options = data_folder
            data_folder = ['']
        else:
            tail_options = ['','data','daf']
            data_folder = [data_folder]
        data_folder = [os.path.join(f,t) for f in data_folder for t in tail_options]
        # look through possibilities until a pfile is found (or not)
    for folder in data_folder:
        for ext in fileExtensions:
            try_filename = fullfile(folder,name,ext)
            if os.path.exists(try_filename):
                return try_filename
    return '' # if no pfile was found

