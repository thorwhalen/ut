"""logging functions"""

__author__ = 'thorwhalen'

from ut.util.ulist import ascertain_list
import logging
from datetime import datetime
# import json
from sys import stdout

default_log_filepath = 'default_log.log'
# a_bunch_of_space = ' ' * 99


def print_progress(msg, refresh=None, display_time=True):
    """
    input: message, and possibly args (to be placed in the message string, sprintf-style
    output: Displays the time (HH:MM:SS), and the message
    use: To be able to track processes (and the time they take)
    """
    if display_time:
        msg = hms_message(msg)
    if refresh is not False:
        print(msg, '\r')
        # stdout.write('\r' + msg)
        # stdout.write(refresh)
        # stdout.flush()
    else:
        print(msg)



def printProgress(message='', args=None, refresh=False, refresh_suffix=None):
    """
    input: message, and possibly args (to be placed in the message string, sprintf-style
    output: Displays the time (HH:MM:SS), and the message
    use: To be able to track processes (and the time they take)
    """
    if args is None:
        args = list()
    else:
        args = ascertain_list(args)
    if len(args) == 0:
        message = message.replace("{", "{{").replace("}", "}}")
    if refresh:
        stdout.write('\r' + hms_message(message.format(*args)))
        if refresh_suffix is not None:
            stdout.write(refresh_suffix)
        stdout.flush()
    else:
        print(hms_message(message.format(*args)))


def hms_message(msg=''):
    t = datetime.now()
    return "({:02.0f}){:02.0f}:{:02.0f}:{:02.0f} - {}".format(t.day, t.hour, t.minute, t.second, msg)


def print_iter_one_per_line(it):
    for x in it:
        print(x)

def get_a_logger(**kwargs):
    kwargs = dict(dict(
            filename=default_log_filepath,
            level='DEBUG',
            format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ), **kwargs)

    root_logger = logging.getLogger()
    root_logger.setLevel(kwargs['level'])

    # setup custom logger
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(kwargs['filename'])
    handler.setLevel(kwargs['level'])
    logger.addHandler(handler)
    return logger