__author__ = 'thor'

from collections import defaultdict


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        ret = self[key] = self.default_factory(key)
        return ret


class DictDefaultDict(dict):
    def __init__(self, default_dict):
        super(DictDefaultDict, self).__init__()
        self.default_dict = default_dict
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            return self.default_dict[item]