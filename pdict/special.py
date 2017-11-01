__author__ = 'thor'

from collections import defaultdict

val_unlikely_to_be_value_of_dict = (1987654321, 8239080923)

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


class KeyPathDict(dict):
    def get(self, key_path, d=None):
        #         return get_value_in_key_path(dict(KeyPathDict), key_path, d)
        if isinstance(key_path, basestring):
            key_path = key_path.split('.')
        k_length = len(key_path)
        if k_length == 0:
            return super(KeyPathDict, self).get(key_path[0], d)
        else:
            val_so_far = super(KeyPathDict, self).get(key_path[0], d)
            for key in key_path[1:]:
                if isinstance(val_so_far, dict):
                    val_so_far = val_so_far.get(key, val_unlikely_to_be_value_of_dict)
                    if val_so_far == val_unlikely_to_be_value_of_dict:
                        return d
                else:
                    return d
            return val_so_far

    def __getitem__(self, val):
        return self.get(val, None)

    def __setitem__(self, key_path, val):
        if isinstance(key_path, basestring):
            key_path = key_path.split('.')
        self[key_path[:-1]].__setitem__(key_path[-1], val)