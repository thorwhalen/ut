__author__ = 'thorwhalen'

# utils to get from a pfile to... something else
from ut.pfile.name import replace_extension
import os
import gzip


def string(filename):
    """
    returns the string contents of a pfile
    """
    fid = file(filename)
    s = fid.read()
    fid.close()
    return s


def zip_file(source_file, destination_file=None):
    if destination_file is None:  # if no destination file is given, add the .zip extension to create the dest file
        destination_file = source_file + '.zip'
        assert destination_file != source_file  # to make sure source and destination are not the same
    elif os.path.isdir(destination_file):
        destination_file = os.path.join(destination_file, source_file + '.zip')
    source_file = source_file.replace('$', '\$') # replacing the unix-escape character $ with \$
    os_system_result = os.system('zip "'+ destination_file.replace('$', '\$') + '" "' + source_file + '"')
    return destination_file


def gzip_file(source_file, destination_file=None):
    import gzip
    if destination_file is None:  # if no destination file is given, add the .zip extension to create the dest file
        destination_file = source_file + '.gzip'
        assert destination_file != source_file  # to make sure source and destination are not the same
    elif os.path.isdir(destination_file):
        destination_file = os.path.join(destination_file, source_file + '.gzip')
    with open(source_file, 'rb') as orig_file:
        with gzip.open(destination_file, 'wb') as zipped_file:
            zipped_file.writelines(orig_file)
    return destination_file


def ungzip(gzip_file, destination_file):
    in_file = gzip.open(gzip_file, 'rb')
    out_file = open(destination_file, 'wb')
    out_file.write(in_file.read())
    in_file.close()
    out_file.close()


def tail(f, window=20):
    """
    Returns the last `window` lines of file `f` as a list.
    """
    if isinstance(f, basestring):
        f = open(f, 'r')
        file_should_be_closed = True
    else:
        file_should_be_closed = False
    if window == 0:
        return []
    BUFSIZ = 1024
    f.seek(0, 2)
    bytes = f.tell()
    size = window + 1
    block = -1
    data = []
    while size > 0 and bytes > 0:
        if bytes - BUFSIZ > 0:
            # Seek back one whole BUFSIZ
            f.seek(block * BUFSIZ, 2)
            # read BUFFER
            data.insert(0, f.read(BUFSIZ))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            data.insert(0, f.read(bytes))
        linesFound = data[0].count('\n')
        size -= linesFound
        bytes -= BUFSIZ
        block -= 1
    if file_should_be_closed:
        f.close()
    return '\n'.join(''.join(data).splitlines()[-window:])