__author__ = 'thorwhalen'

from ut.util.ulist import ascertain_list
import logging
from datetime import datetime
# import json
from sys import stdout

default_log_filepath = 'default_log.log'


def printProgress(message='', args=[], refresh=False):
    """
    input: message, and possibly args (to be placed in the message string, sprintf-style
    output: Displays the time (HH:MM:SS), and the message
    use: To be able to track processes (and the time they take)
    """
    args = ascertain_list(args)
    if refresh:
        stdout.write('\r' + hms_message(message.format(*args)))
        stdout.flush()
    else:
        print(hms_message(message.format(*args)))


def hms_message(msg=''):
    t = datetime.now()
    return "({:02.0f}){:02.0f}:{:02.0f}:{:02.0f} - {}".format(t.day, t.hour, t.minute, t.second, msg)


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