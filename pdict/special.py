__author__ = 'thor'


class DictDefaultDict(dict):
    def __init__(self, default_dict):
        super(DictDefaultDict, self).__init__()
        self.default_dict = default_dict
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            return self.default_dict[item]