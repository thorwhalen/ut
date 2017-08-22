from __future__ import division


class Buffer(object):
    def __init__(self):
        """
        Base class for the buffer pattern.
        The interface methods are
            * push(item): To push an item to the buffer, and flushing if the should_flush() condition is met.
            * flush(): Flush the buffer
            * should_flush(): Condition deciding whether to flush or not (usually depends on the buf_info()
            * iterate(iterator): Loop through items of iterator, and call push(item) for every item. Additionally,
                call flush() when the loop is completed. Return number of items encountered.
            * buf_info: Information about the buffer
        The hidden (but defining) attributes:
            * _buf: The object holding the buffer
            * _push: The operation of pushing an item to the buffer (without flushing)
            * _flush: The operation of flushibng the buffer (without reinitializing it)

        Though meant to be an interface class, all methods have been implemented.
        They all assume that _buf is a list. _push(item) simply appends to the list and _flush simply returns the list.
        Additionally buf_info() is the length of the list and should_flush() is always true.
        """
        self.initialize()

    def initialize(self):
        self._buf = list()

    def _push(self, item):
        self._buf.append(item)

    def _flush(self):
        return self._buf

    def push(self, item):
        """
        Push item in the buffer (using self._push(item)), and if self.should_flush() is True,
        call self._flush() and self.initialize()
        :param item: The item to push
        :return: The return value of self._flush() if flushed, and None otherwise.
        """
        self._push(item)
        if self.should_flush():
            return self.flush()

    def flush(self):
        r = self._flush()
        self.initialize()
        return r

    def buf_info(self):
        return self._buf.__len__()

    def should_flush(self):
        """
        Condition deciding whether to flush or not.
        :return:
        """
        return True

    def iterate(self, iterator):
        """
        Iterate over items, pushing every item, and call flush when iterator consumed.
        :param iterator: item iterator
        :return: The number of items consumed from the iterator
        """
        n_ops = 0
        for n_ops, item in enumerate(iterator, 1):
            self.push(item)
        self.flush()
        return n_ops


class FixedSizeBuffer(Buffer):
    def __init__(self, max_buf_size):
        """

        :param max_buf_size:

        >>> from numpy import sum
        >>> class SumFixedSizeBuffer(FixedSizeBuffer):
        ...     def _flush(self):
        ...         val = sum(self._buf)
        ...         print("SumFixedSizeBuffer._flush({}): {}".format(self.max_buf_size, val))
        ...         return sum(val)
        ...
        >>> b = SumFixedSizeBuffer(max_buf_size=3)
        >>> for item in range(5):
        ...     print('----> {}'.format(item))
        ...     b.push(item)
        ----> 0
        ----> 1
        ----> 2
        SumFixedSizeBuffer._flush(3): 3
        3
        ----> 3
        ----> 4
        >>> b.flush()
        SumFixedSizeBuffer._flush(3): 7
        7
        """
        super(FixedSizeBuffer, self).__init__()
        self.max_buf_size = max_buf_size

    def should_flush(self):
        return self.buf_info() >= self.max_buf_size
