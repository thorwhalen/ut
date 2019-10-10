"""
    An example of connecting the flush of one buffer to the push of another.

    >>> from ut.dpat.buffer import ThreshBuffer, inject_post_flush_func
    >>> class SumFixedSizeBuffer(ThreshBuffer):
    ...     def _flush(self):
    ...         return sum(self._buf)
    ...
    >>> buf_a = SumFixedSizeBuffer(thresh=2)
    >>> buf_b = SumFixedSizeBuffer(thresh=3)
    >>> buf_a = inject_post_flush_func(buf_a, buf_b.push)
    >>> for i in range(13):
    ...     print("push: {}".format(i))
    ...     print("  push output: {}".format(buf_a.push(i)))
    ...     print("  from_buf._buf: {}".format(buf_a._buf))
    ...     print("    to_buf._buf: {}".format(buf_b._buf))
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
    >>> buf_a.flush()
    >>> print("flush_all() output: {}".format(buf_b.flush()))
    flush_all() output: 12
    """



from collections import defaultdict


def _add_pre_input_func(method, pre_input_func):
    def wrapper(*args):
        return method(pre_input_func(*args))

    return wrapper


def _add_post_output_func(method, post_output_func):
    def wrapper(*args):
        return post_output_func(method(*args))

    return wrapper


def inject_pre_push_func(buf, pre_push_func):
    buf.push = _add_pre_input_func(buf.push, pre_push_func)
    return buf


def inject_post_flush_func(buf, post_flush_func):
    def post_flush_func_with_not_none(x):
        if x is not None:
            return post_flush_func(x)

    buf.flush = _add_post_output_func(buf.flush, post_flush_func_with_not_none)
    return buf


def print_buf_pre_push(buf):
    def print_buf(item):
        print((buf._buf))
        return item

    return inject_pre_push_func(buf, print_buf)


def print_flush(buf):
    def print_flush(item):
        print(item)
        return item

    return inject_post_flush_func(buf, print_flush)

def reroute_flush_to_push(from_buf, to_buf):
    inject_post_flush_func(from_buf, to_buf.push)


def mk_fixed_size_buffer(max_buf_size):
    return ThreshBuffer(thresh=max_buf_size)


def name_buffer(buffer, name):
    setattr(buffer, 'name', name)
    return buffer


class AbstractBuffer(object):
    """
    An abstract buffer just specifies that there needs to be a push and a flush method
    """

    def __call__(self, item):
        if item is not None:
            return self.push(item)

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


# def wrap_push_and_flush(buffer_class, pre_push=None, post_flush=None, *args, **kwargs):
#     class BufferWrap


class Router(AbstractBuffer):
    def __init__(self, buf_list):
        """
        A (buffer) router holds a list of Buffer objects (in buf_list), and
            * when pushing an item, loops through the buffers of the list and pushes the item to each
            * when flushing, loops through the buffers of the list and flushes each
        :param buf_list: A list of Buffer objects

        Note: The flush of a Router buffer returns None. This means they are not meant for flushing with output.

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


class WrappedRouter(AbstractBuffer):
    def __init__(self, buf_list):
        self.buf_list = buf_list
        for i, buf_spec in enumerate(self.buf_list):
            if isinstance(buf_spec, dict):
                buf_spec['pre_push'] = buf_spec.get('pre_push', lambda x: x)
                buf_spec['post_flush'] = buf_spec.get('post_flush', lambda x: x)
            else:
                self.buf_list[i] = {
                    'buf': self.buf_list[i],
                    'pre_push': lambda x: x,
                    'post_flush': lambda x: x
                }

    def push(self, item):
        """
        Call buf.push(pre_push(item)) for every buf of buf_list.
        :param item:
        :return: None
        """
        for buf_spec in self.buf_list:
            buf_spec['push'].push(buf_spec['pre_push'](item))

    def flush(self):
        """
        Flushes all buffers in object's buf_list, feeding them to their post_flush function.
        :return: None
        """
        for buf_spec in self.buf_list:
            buf_spec['post_flush'](buf_spec['buf'].flush())


class BufferPipeline(AbstractBuffer):
    """
    A buffer made by composing a list of buffers.
    Pushing an item to this buffer will push to the first buffer of the list, but if and when it flushes, the return
    value of the flush will be pushed to the next buffer, and so on through the list.
    Flushing will result in flushing every buffer of the list, in order. Again, flush output of buffer[i] is pushed to
    buffer[i+1], which is then force-flushed, and so on. The output of this pipeline flush will be the output of the
    last buffer of the pipeline, which will have integrated the flushes of all previous buffers.

    >>> from ut.dpat.buffer import ThreshBuffer, BufferPipeline
    >>> class SumFixedSizeBuffer(ThreshBuffer):
    ...     def _flush(self):
    ...         return sum(self._buf)
    ...
    >>> class SumFixedSizeBufferWithPrint(ThreshBuffer):
    ...     def _flush(self):
    ...         r = sum(self._buf)
    ...         print("  SumFixedSizeBufferWithPrint: {}".format(r))
    ...         return r
    ...
    >>> buf_a = SumFixedSizeBuffer(thresh=2)
    >>> buf_b = SumFixedSizeBuffer(thresh=3)
    >>> buf_c = SumFixedSizeBufferWithPrint(thresh=2)
    >>> b = BufferPipeline(buf_list=[buf_a, buf_b, buf_c])
    >>> for item in range(16):
    ...     print("push: {}".format(item))
    ...     print("  push output: {}".format(b.push(item)))
    ...     for j in range(len(b.buf_list)):
    ...         print("     buf[{}]._buf: {}".format(j, b.buf_list[j]._buf))
    ...
    push: 0
      push output: None
         buf[0]._buf: [0]
         buf[1]._buf: []
         buf[2]._buf: []
    push: 1
      push output: None
         buf[0]._buf: []
         buf[1]._buf: [1]
         buf[2]._buf: []
    push: 2
      push output: None
         buf[0]._buf: [2]
         buf[1]._buf: [1]
         buf[2]._buf: []
    push: 3
      push output: None
         buf[0]._buf: []
         buf[1]._buf: [1, 5]
         buf[2]._buf: []
    push: 4
      push output: None
         buf[0]._buf: [4]
         buf[1]._buf: [1, 5]
         buf[2]._buf: []
    push: 5
      push output: None
         buf[0]._buf: []
         buf[1]._buf: []
         buf[2]._buf: [15]
    push: 6
      push output: None
         buf[0]._buf: [6]
         buf[1]._buf: []
         buf[2]._buf: [15]
    push: 7
      push output: None
         buf[0]._buf: []
         buf[1]._buf: [13]
         buf[2]._buf: [15]
    push: 8
      push output: None
         buf[0]._buf: [8]
         buf[1]._buf: [13]
         buf[2]._buf: [15]
    push: 9
      push output: None
         buf[0]._buf: []
         buf[1]._buf: [13, 17]
         buf[2]._buf: [15]
    push: 10
      push output: None
         buf[0]._buf: [10]
         buf[1]._buf: [13, 17]
         buf[2]._buf: [15]
    push: 11
      SumFixedSizeBufferWithPrint: 66
      push output: 66
         buf[0]._buf: []
         buf[1]._buf: []
         buf[2]._buf: []
    push: 12
      push output: None
         buf[0]._buf: [12]
         buf[1]._buf: []
         buf[2]._buf: []
    push: 13
      push output: None
         buf[0]._buf: []
         buf[1]._buf: [25]
         buf[2]._buf: []
    push: 14
      push output: None
         buf[0]._buf: [14]
         buf[1]._buf: [25]
         buf[2]._buf: []
    push: 15
      push output: None
         buf[0]._buf: []
         buf[1]._buf: [25, 29]
         buf[2]._buf: []
    >>> print("flush_all() output: {}".format(b.flush()))
      SumFixedSizeBufferWithPrint: 54
    flush_all() output: 54
    """

    def __init__(self, buf_list):
        """
        :param buf_list: A list of Buffer objects.
        """
        self.buf_list = buf_list
        for i in range(len(self.buf_list) - 1):
            self.buf_list[i] = inject_post_flush_func(self.buf_list[i], self.buf_list[i + 1].push)

    def push(self, item):
        if item is not None:
            return self.buf_list[0].push(item)
        # for buf in self.buf_list:
        #     item = buf.push(item)
        #     if item is None:
        #         return None
        # return item

    def flush(self):
        for i in range(len(self.buf_list) - 1):
            self.buf_list[i].flush()
        return self.buf_list[-1].flush()
        # item = None
        # for buf in self.buf_list:
        #     if item is not None:
        #         buf.push(item)
        #     item = buf.flush()
        # return item


class BufferWrapper(AbstractBuffer):
    def __init__(self, buf):
        self.buf = buf

    def push(self, item):
        if item is not None:
            return self.buf.push(item)

    def flush(self):
        return self.buf.flush()

    def __getattr__(self, attr):
        if attr not in ['push', 'flush']:
            return getattr(self.buf, attr)
        else:
            return getattr(self, attr)


class TransparentBuffer(AbstractBuffer):
    def __init__(self):
        self.initialize()

    def initialize(self):
        self._buf = None

    def push(self, item):
        if item is not None:
            self._buf = item

    def flush(self):
        return self._buf


class PassThroughBuffer(AbstractBuffer):
    def __init__(self, pre_push_funcs=(), post_flush_funcs=()):
        self.pre_push_funcs = pre_push_funcs
        self.post_flush_funcs = post_flush_funcs
        self.initialize()

    def initialize(self):
        self._buf = None

    def push(self, item):
        for func in self.pre_push_funcs:
            item = func(item)
        self._buf = item

    def flush(self):
        item = self._buf
        if item is not None:
            for func in self.post_flush_funcs:
                item = func(item)
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
        if item is not None:
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
    >>> import operator
    >>> print('final flush: {}'.format(sorted(b.flush_all().items(),key=operator.itemgetter(0))))
    final flush: [(30, [{'x': 'bar', 't': 30}]), (40, [{'x': '!', 't': 43}])]
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
        Initialize _buf. Here this means poping off the lowest key, if there's any.
        :return:
        """
        if len(self._buf) > 0:
            min_key = min(self._buf.keys())
            self._buf.pop(min_key)

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
        If len(_buf) >= thresh, return all elements of _buf BUT the element with the highest
        :return:
        """
        if len(self._buf) > 0:
            min_key = min(self._buf.keys())
            return {min_key: self._buf[min_key]}
            # return {min_key: self._buf.pop(min_key)}
        else:
            return self._buf

    def flush_all(self):
        r = dict(self._buf)
        self._buf = defaultdict(list)
        return r
