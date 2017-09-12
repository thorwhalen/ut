__author__ = 'thor'

import os
import errno


def dir_but_not_ancestors(newdir):
    """ make the directory newdir, except if
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory doesn't exist, raise an exception
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " \
                      "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            raise OSError("The parent directory %s doesn't exist, so you can't make %s in it " % (head, tail))
        elif tail:
            os.mkdir(newdir)


def create_missing_directory_for_filename(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise