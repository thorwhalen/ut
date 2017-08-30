from __future__ import division

from collections import defaultdict
from ut.util.pobj import inject_method


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

    def flush_all(self):
        """
        Flush all contents of _buf.
        Used at the end of an iterator to make sure we get all the data, regardless of the should_flush() condition.
        By default just calls flush() but in some cases (where flush doesn't empty the _buf completely),
        should be overwritten to actually flush all data.
        """
        return self.flush()


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


class BufferLink(AbstractBuffer):
    """
    An AbstractBuffer obtained by composing two buffers: from_buf and to_buf.
    When push(item) is called, it is passed on to from_buf.push(item).
    If that later returns something (because it was flushed), the output will be pushed to to_buf
    (i.e. to_buf.push(output_of_from_buf_push).

    Note: For proper use, the from_buf.push should only return an output whose bool(output) resolves to True if and only
    if this output is meant to be passed on as an item in a to_buf.push(item)

    By composing several BufferLinks together, we can get pipelines and buffer computation DAGs.

    >>> from ut.dpat.buffer import ThreshBuffer, BufferLink
    >>> class SumFixedSizeBuffer(ThreshBuffer):
    ...     def _flush(self):
    ...         return sum(self._buf)
    ...
    >>> buf_a = SumFixedSizeBuffer(thresh=2)
    >>> buf_b = SumFixedSizeBuffer(thresh=3)
    >>> b = BufferLink(from_buf=buf_a, to_buf=buf_b)
    >>> for i in range(13):
    ...     print("push: {}".format(i))
    ...     print("  push output: {}".format(b.push(i)))
    ...     print("  from_buf._buf: {}".format(b.from_buf._buf))
    ...     print("    to_buf._buf: {}".format(b.to_buf._buf))
    ...
    push: 0
      push output: None
      from_buf._buf: [0]
        to_buf._buf: []
    push: 1
      push output: None
      from_buf._buf: []
        to_buf._buf: [1]
    push: 2
      push output: None
      from_buf._buf: [2]
        to_buf._buf: [1]
    push: 3
      push output: None
      from_buf._buf: []
        to_buf._buf: [1, 5]
    push: 4
      push output: None
      from_buf._buf: [4]
        to_buf._buf: [1, 5]
    push: 5
      push output: 15
      from_buf._buf: []
        to_buf._buf: []
    push: 6
      push output: None
      from_buf._buf: [6]
        to_buf._buf: []
    push: 7
      push output: None
      from_buf._buf: []
        to_buf._buf: [13]
    push: 8
      push output: None
      from_buf._buf: [8]
        to_buf._buf: [13]
    push: 9
      push output: None
      from_buf._buf: []
        to_buf._buf: [13, 17]
    push: 10
      push output: None
      from_buf._buf: [10]
        to_buf._buf: [13, 17]
    push: 11
      push output: 51
      from_buf._buf: []
        to_buf._buf: []
    push: 12
      push output: None
      from_buf._buf: [12]
        to_buf._buf: []
    >>>
    >>> print("flush_all() output: {}".format(b.flush_all()))
    flush_all() output: 12
    """

    def __init__(self, from_buf, to_buf):
        self.from_buf = from_buf
        self.to_buf = to_buf
        self.initialize()

    def initialize(self):
        self.from_buf.initialize()
        self.to_buf.initialize()

    def push(self, item):
        r = self.from_buf.push(item)
        if r:
            return self.to_buf.push(r)

    def flush(self):
        """
        Flushes all buffers in object's buf_list.
        :return: None
        """
        r = self.from_buf.flush()
        if r:
            self.to_buf.push(r)
        return self.to_buf.flush()


def reroute_flush(from_buf, to_buf):
    def flush():
        r = from_buf.flush()
        if r:
            return to_buf.push(r)

    return flush


class BufferPipeline(AbstractBuffer):
    def __init__(self, buf_list):
        self.buf_list = buf_list
        self.n_buffers = len(self.buf_list)
        # self.buf_link = list()
        # pipeline_buf = BufferLink(from_buf=self.buf_list[-2], to_buf=self.buf_list[-1])
        # for i in range(self.n_buffers - 1)[::-1][1:]:
        #     from_buf = self.buf_list[i]
        #     pipeline_buf = BufferLink(from_buf=self.buf_list[i], to_buf=pipeline_buf)
        #     # from_buf = self.buf_list[i]
        #     # to_buf = self.buf_list[i + 1]
        #     # from_buf.flush = inject_method(from_buf, )
        #     self.buf_link.append(BufferLink(from_buf=self.buf_list[i], to_buf=self.buf_list[i + 1]))

    def push(self, item):
        for buf in self.buf_list:
            item = buf.push(item)
            if item is None:
                return None
        return item

    def flush(self):
        item = None
        for buf in self.buf_list:
            if item is not None:
                buf.push(item)
            item = buf.flush()
        return item


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
        # if self.should_flush():
        #     r = self.flush()
        # else:
        #     r = None
        # self._push(item)
        # return r
        self._push(item)
        if self.should_flush():
            return self.flush()

    def flush(self):
        r = self._flush()
        self.initialize()
        if r:
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
        Note that not flush() or flush_all() outputs will but returned.
        This is meant only to be used if flush and flush_all do their work inplace (such as writing to a db).
        :param iterator: item iterator
        :return: The number of items consumed from the iterator
        """
        n_ops = 0
        for n_ops, item in enumerate(iterator, 1):
            self.push(item)
        self.flush_all()
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


class BucketBuffer(ThreshBuffer):
    """
    A ThreshBuffer that maintains a dict buffer where items, when pushed, are pushed to different keys.
    The key where item will be stored is defined by the _buf_key_for_item(item) method.
    The _buf_key_for_item(item) is (unless overwrittern) defined by the index_field and bucket_step as follows:
        _buf_key_for_item(item) = bucket_step * (item[index_field] // bucket_step)
    That is, the method will take the maximum multiple of bucket_step that is no greater than the item[index_field] val.

    This is useful, for instance, to accumulate data according to fixed sized time segments.

    items pushed must contain a item[index_field] element. Usually items are dicts.

    >>> from ut.dpat.buffer import BucketBuffer
    >>>
    >>> input_dicts = [
    ...     {'t': 0, 'x': 'hello'},
    ...     {'t': 1, 'x': 'world'},
    ...     {'t': 12, 'x': 'this'},
    ...     {'t': 19, 'x': 'is'},
    ...     {'t': 24, 'x': 'foo'},
    ...     {'t': 30, 'x': 'bar'},
    ...     {'t': 43, 'x': '!'}
    ... ]
    >>> b = BucketBuffer(thresh=3, index_field='t', bucket_step=10)
    >>> for d in input_dicts:
    ...     print('push: {}'.format(d))
    ...     r = b.push(d)
    ...     if isinstance(r, dict):
    ...         r = dict(r)
    ...     print('flushed: {}'.format(r))
    ...     print('  _buf: {}'.format(dict(b._buf)))
    ...
    push: {'x': 'hello', 't': 0}
    flushed: None
      _buf: {0: [{'x': 'hello', 't': 0}]}
    push: {'x': 'world', 't': 1}
    flushed: None
      _buf: {0: [{'x': 'hello', 't': 0}, {'x': 'world', 't': 1}]}
    push: {'x': 'this', 't': 12}
    flushed: None
      _buf: {0: [{'x': 'hello', 't': 0}, {'x': 'world', 't': 1}], 10: [{'x': 'this', 't': 12}]}
    push: {'x': 'is', 't': 19}
    flushed: None
      _buf: {0: [{'x': 'hello', 't': 0}, {'x': 'world', 't': 1}], 10: [{'x': 'this', 't': 12}, {'x': 'is', 't': 19}]}
    push: {'x': 'foo', 't': 24}
    flushed: {0: [{'x': 'hello', 't': 0}, {'x': 'world', 't': 1}]}
      _buf: {10: [{'x': 'this', 't': 12}, {'x': 'is', 't': 19}], 20: [{'x': 'foo', 't': 24}]}
    push: {'x': 'bar', 't': 30}
    flushed: {10: [{'x': 'this', 't': 12}, {'x': 'is', 't': 19}]}
      _buf: {20: [{'x': 'foo', 't': 24}], 30: [{'x': 'bar', 't': 30}]}
    push: {'x': '!', 't': 43}
    flushed: {20: [{'x': 'foo', 't': 24}]}
      _buf: {40: [{'x': '!', 't': 43}], 30: [{'x': 'bar', 't': 30}]}
    >>> print('final flush: {}'.format(b.flush_all()))
    final flush: [{30: [{'x': 'bar', 't': 30}]}, {40: [{'x': '!', 't': 43}]}]
    """

    def __init__(self, thresh, index_field, bucket_step):
        """
        :param thresh: Used as in ThreshBuffer. Once the length of _buf is equal or greater to this value, the _buf
            will be flushed. Note though that _buf is a dict (a defaultdict(list) to be exact), so the length is the
            number of keys. Setting thresh to 2 will have the effect of flushing as soon as data in a new bucket is
            pushed. This will probably only have the desired effect if the data arrives with the index (in index_field)
            sorted.
        :param index_field: The key of the item to get the value to be "bucketed"
        :param bucket_step: The integer to divide the item[index_field] value by to determine the bucket it should be
            stored in.
        """
        assert isinstance(thresh, int), "thresh must be an int"
        assert isinstance(bucket_step, int), "bucket_step must be an int"
        self.index_field = index_field
        self.bucket_step = bucket_step
        self._buf = defaultdict(list)
        super(BucketBuffer, self).__init__(thresh=thresh)

    def initialize(self):
        """
        Initialize _buf with self._initialize_buf_with (originally an empty dict)
        :return:
        """
        pass
        # self._buf = defaultdict(list, **self._initialize_buf_with)

    def _buf_key_for_item(self, item):
        """
        Returns the key of _buf where the item should be stored.
        :param item: A dict. Must have the self.index_field key
        :return: A (int) bucket index
        """
        return self.bucket_step * (item[self.index_field] // self.bucket_step)

    def _push(self, item):
        """
        Appends item to a bucket of the _buf.
        _buf is a dict keyed by
        :param item:
        :return:
        """
        self._buf[self._buf_key_for_item(item)].append(item)

    def _flush(self):
        """
        If len(_buf) > 1, return all elements of _buf BUT the element with the highest
        :return:
        """
        if len(self._buf) > 0:
            min_key = min(self._buf.keys())
            return {min_key: self._buf.pop(min_key)}
        else:
            return self._buf

    def flush_all(self):
        r = list()
        while len(self._buf) > 0:
            r.append(self._flush())
        return r

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
