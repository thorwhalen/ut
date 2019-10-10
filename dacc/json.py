__author__ = 'thor'

from . import json


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    Enabling json dumps of objects containing numpy arrays.
    Usage:
        json.dumps(obj, cls=NumpyAwareJSONEncoder)
    """
    def default(self, obj):
        try:
            json.JSONEncoder.default(self, obj)
        except TypeError:
            return list(obj)

