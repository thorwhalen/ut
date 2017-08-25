from __future__ import division


def mk_fixed_size_buffer(max_buf_size):
    return ThreshBuffer(thresh=max_buf_size)


class AbstractBuffer(object):
    """
    An abstract buffer just specifies that there needs to be a push and a flush method
    """

    def push(self, item):
        raise NotImplementedError("Needs to be implemented in a concrete class.")

    def flush(self):
        raise NotImplementedError("Needs to be implemented in a concrete class.")


class Router(AbstractBuffer):
    def __init__(self, buf_list):
        """
        A (buffer) router holds a list of Buffer objects (in buf_list), and
            * when pushing an item, loops through the buffers of the list and pushes the item to each
            * when flushing, loops through the buffers of the list and flushes each
        :param buf_list: A list of Buffer objects

        >>> from numpy import sum
        >>> class StoreBuffer(ThreshBuffer):
        ...     def _flush(self):
        ...         print("  StoreBuffer: {}".format(self._buf))
        ...         super(StoreBuffer, self)._flush()
        >>> class AggrBuffer(ThreshBuffer):
        ...     def _flush(self):
        ...         t = sum(self._buf)
        ...         print("  AggrBuffer: {}".format(t))
        ...         super(AggrBuffer, self)._flush()
        ...
        >>>
        >>> router = Router([StoreBuffer(3), AggrBuffer(2)])
        >>> for i in range(10):
        ...     print("item: {}".format(i))
        ...     router.push(i)
        ...
        item: 0
        item: 1
          AggrBuffer: 1
        item: 2
          StoreBuffer: [0, 1, 2]
        item: 3
          AggrBuffer: 5
        item: 4
        item: 5
          StoreBuffer: [3, 4, 5]
          AggrBuffer: 9
        item: 6
        item: 7
          AggrBuffer: 13
        item: 8
          StoreBuffer: [6, 7, 8]
        item: 9
          AggrBuffer: 17
        """
        self.buf_list = buf_list

    def push(self, item):
        """
        Call buf.push(item) for every buf of buf_list.
        :param item:
        :return: None
        """
        for buf in self.buf_list:
            buf.push(item)

    def flush(self):
        """
        Flushes all buffers in object's buf_list.
        :return: None
        """
        for buf in self.buf_list:
            buf.flush()


class Buffer(AbstractBuffer):
    def __init__(self):
        """
        Base class for the buffer pattern.
        The interface methods are
            * push(item): To push an item to the buffer, and flushing if the should_flush() condition is met.
            * flush(): Flush the buffer
            * should_flush(): Condition deciding whether to flush or not
            * iterate(iterator): Loop through items of iterator, and call push(item) for every item. Additionally,
                call flush() when the loop is completed. Return number of items encountered.
        The hidden (but defining) attributes:
            * _buf: The object holding the buffer
            * _push: The operation of pushing an item to the buffer (without flushing)
            * _flush: The operation of flushibng the buffer (without reinitializing it)

        Though meant to be an interface class, all methods have been implemented.
        They all assume that _buf is a list. _push(item) simply appends to the list and _flush simply returns the list.
        Here the should_flush() is implemented to always return True (but in must cases should be overwritten).
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


class ThreshBuffer(Buffer):
    """
    A ThreshBuffer is a buffer that flushes when self.buf_val_for_thresh() >= self.thresh

    * buf_val_for_thresh: Method returning a numerical value for self._buf that will be used for the >= self.thresh
        comparison. By default, self._buf.__len__() is used.

    :param thresh: The threshold that determines when to flush.

    >>> b = ThreshBuffer(thresh=3)
    >>> for item in range(5):
    ...     print b.push(item)
    ...
    None
    None
    [0, 1, 2]
    None
    None
    >>> b.flush()
    [3, 4]
    >>>
    >>> from numpy import sum
    >>> class SumFixedSizeBuffer(ThreshBuffer):
    ...     def _flush(self):
    ...         val = sum(self._buf)
    ...         print("SumFixedSizeBuffer._flush({}): {}".format(self.thresh, val))
    ...         return sum(val)
    ...
    >>> b = SumFixedSizeBuffer(thresh=3)
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

    def __init__(self, thresh):
        """
        :param thresh: The threshold that determines when to flush.
        """

        super(ThreshBuffer, self).__init__()
        self.thresh = thresh

    def buf_val_for_thresh(self):
        return self._buf.__len__()

    def should_flush(self):
        return self.buf_val_for_thresh() >= self.thresh


class TimeBucketBuffer(ThreshBuffer):
    def __init__(self, thresh, time_field):
        super(TimeBucketBuffer, self).__init__(thresh=thresh)
        assert isinstance(thresh, int), "time must be expressed as integers"
        self.time_field = time_field

    def _tick_for(self, ts):
        return ts // self.thresh

    def buf_val_for_thresh(self):
        return self._last_tick_flushed()
        # self._last_thresh

    def _push(self, item):
        self._last_tick = item[self.time_field]
        return super(TimeBucketBuffer, self)._push(item)

    def _flush(self):
        self._last_tick_flushed = self._tick_for(self._last_time)



# class FixedSizeBuffer(ThreshBuffer):
#     def __init__(self, max_buf_size):
#         """
#         A ThreshBuffer that is meant to be used with a _buf list whose size determines when to flush.
#         :param max_buf_size: maximum size of len(_buf) over which to call flush()
#
#         >>> from numpy import sum
#         >>> class SumFixedSizeBuffer(FixedSizeBuffer):
#         ...     def _flush(self):
#         ...         val = sum(self._buf)
#         ...         print("SumFixedSizeBuffer._flush({}): {}".format(self.thresh, val))
#         ...         return sum(val)
#         ...
#         >>> b = SumFixedSizeBuffer(max_buf_size=3)
#         >>> for item in range(5):
#         ...     print('----> {}'.format(item))
#         ...     b.push(item)
#         ----> 0
#         ----> 1
#         ----> 2
#         SumFixedSizeBuffer._flush(3): 3
#         3
#         ----> 3
#         ----> 4
#         >>> b.flush()
#         SumFixedSizeBuffer._flush(3): 7
#         7
#         """
#         super(FixedSizeBuffer, self).__init__(thresh=max_buf_size)
