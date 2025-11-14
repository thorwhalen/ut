__author__ = 'thor'


from collections import deque, Counter


class WindowCollector(deque):
    """
    Collects values (through append) in a limited deque (like a list), starting from the right, and sliding values
    towards the left, until they fall out of the window.
    Note: Convenience class. Know that it is three times slower than doing a popleft() and append(val)
    directly on a deque.

    >>> from ut.pcoll.util import WindowCollector
    >>> t = WindowCollector(3, None)
    >>> t
    deque([None, None, None])
    >>> t.append(1)
    >>> t
    deque([None, None, 1])
    >>> t.append(2)
    >>> t
    deque([None, 1, 2])
    >>> t.append(3)
    >>> t
    deque([1, 2, 3])
    >>> t.append(4)
    >>> t
    deque([2, 3, 4])
    """

    def __init__(self, window_width, empty_val=None):
        super().__init__([empty_val] * window_width)

    def append(self, val):
        self.popleft()
        super().append(val)


def duplicated_values(arr):
    return [x for x, y in list(Counter(arr).items()) if y > 1]
