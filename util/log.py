__author__ = 'thorwhalen'

from ut.util.ulist import ascertain_list
import logging
from datetime import datetime
import json

default_log_filepath = 'default_log.log'


def printProgress(message='', args=[]):
    """
    input: message, and possibly args (to be placed in the message string, sprintf-style
    output: Displays the time (HH:MM:SS), and the message
    use: To be able to track processes (and the time they take)
    """
    args = ascertain_list(args)
    print(hms_message(message.format(*args)))
    # t = datetime.now().time()
    # print("{:.0f}:{:.0f}:{:.0f} - {}".format(t.hour, t.minute, t.second, message.format(*args)))
    # print "%02.0f:%02.0f:%02.0f " % (t.hour, t.minute, t.second) + message.format(*args)

    #def printProgress(message,args):
    #    print "".format([message,pstr(datetime.now().time())]+args)


def hms_message(msg=''):
    t = datetime.now().time()
    return "{:.0f}:{:.0f}:{:.0f} - {}".format(t.hour, t.minute, t.second, msg)
    # return "{:02}:{:02}:{:02} - {}".format(t.hour, t.minute, t.second, msg)


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