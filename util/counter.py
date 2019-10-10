__author__ = 'thorwhalen'


class Counter(object):
    def __init__(self,start=1):
        self.count = start - 1
    def __next__(self):
        self.count += 1
        return self.count


