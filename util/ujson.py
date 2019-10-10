

import json
import traceback

__author__ = 'thor'


def get_dict_of_error_object(error_object, with_traceback=True):

    msg_dict = {'error': str(error_object.__class__), 'message': error_object.args[0]}
    if with_traceback:
        msg_dict['traceback'] = traceback.format_exc()


def get_json_of_error_object(error_object):
    return json.dumps(get_dict_of_error_object(error_object))