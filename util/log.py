__author__ = 'thorwhalen'

from ut.util.ulist import ascertain_list

from datetime import datetime


def printProgress(message='', args=[]):
    """
    input: message, and possibly args (to be placed in the message string, sprintf-style
    output: Displays the time (HH:MM:SS), and the message
    use: To be able to track processes (and the time they take)
    """
    args = ascertain_list(args)
    t = datetime.now().time()
    print "%02.0f:%02.0f:%02.0f " % (t.hour, t.minute, t.second) + message.format(*args)

    #def printProgress(message,args):
    #    print "".format([message,pstr(datetime.now().time())]+args)


def hms_message(msg=''):
    t = datetime.now().time()
    return "{:02}:{:02}:{:02} - {}".format(t.hour, t.minute, t.second, msg)