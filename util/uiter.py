import itertools

# Some of these recipes were taken from
#   https://docs.python.org/2/library/itertools.html#recipes
#   or random other places.
# plus my own sh** and adaptation


from numpy import mod, ndarray, array, floor, inf
from collections import deque
from datetime import datetime
from itertools import islice, chain, combinations, zip_longest
from operator import itemgetter, is_not
import operator
from functools import partial

from random import random
from numpy import floor

is_not_none = partial(is_not, None)


def running_mean(it, chk_size=2, chk_step=1):  # TODO: A version of this with chk_step as well
    """
    Running mean (moving average) on iterator.
    Note: When input it is list-like, ut.stats.smooth.sliders version of running_mean is 4 times more efficient with
    big (but not too big, because happens in RAM) inputs.
    :param it: iterable
    :param chk_size: width of the window to take means from
    :return:

    >>> list(running_mean([1, 3, 5, 7, 9], 2))
    [2.0, 4.0, 6.0, 8.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 2, chk_step=2))
    [2.0, 6.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 2, chk_step=3))
    [2.0, 8.0]
    >>> list(running_mean([1, 3, 5, 7, 9], 3))
    [3.0, 5.0, 7.0]
    >>> list(running_mean([1, -1, 1, -1], 2))
    [0.0, 0.0, 0.0]
    >>> list(running_mean([-1, -2, -3, -4], 3))
    [-2.0, -3.0]
    """

    if chk_step > 1:
        # TODO: perhaps there's a more efficient way. A way that would sum the values of every step and add them in bulk
        yield from itertools.islice(running_mean(it, chk_size), None, None, chk_step)
    else:
        it = iter(it)
        if chk_size > 1:

            c = 0
            fifo = deque([], maxlen=chk_size)
            for i, x in enumerate(it, 1):
                fifo.append(x)
                c += x
                if i >= chk_size:
                    break

            yield c / chk_size

            if chk_step == 1:
                for x in it:
                    c += x - fifo[0]  # NOTE: seems faster than fifo.popleft
                    fifo.append(x)
                    yield c / chk_size
            else:
                raise RuntimeError("This should really never happen, by design.")
                # Below was an attempt at a faster solution than using the islice as is done above.
                # raise NotImplementedError("Not yet implemented (correctly)")
                # for chk in chunker(it, chk_size=chk_size, chk_step=chk_step, return_tail=False):
                #     print(chk)
                #     for x in chk:
                #         c += x - fifo.popleft()
                #     fifo.extend(chk)
                #     yield c / chk_size

        else:
            for x in it:
                yield x


def _inefficient_indexed_sliding_window_chunk_iter(it, chk_size, chk_step=None,
                                                   start_at=None, stop_at=None, key=None, return_tail=True):
    """
    a function to get (an iterator of) segments (bt, tt) of chunks from (an iterator of) ordered timestamps,
    given a chk_size, chk_step, and a start_at time.
    :param it:
    :param chk_size:
    :param chk_step:
    :param start_at:
    :param key:
    :param return_tail:
    :return:
    """
    if chk_step is None:
        chk_step = chk_size
    if key is None:
        key = lambda x: x
    it = array(list(it))
    it_key = array(list(map(key, it)))
    assert all(sorted(it_key) == it_key), 'iterator was not sorted'
    if start_at is None:
        start_at = it_key[0]
    if stop_at is None:
        stop_at = it_key[-1]

    bt = start_at
    tt = start_at + chk_size
    while bt < stop_at:
        lidx = it_key >= bt
        lidx &= it_key < tt
        yield list(it[lidx])
        bt += chk_step
        tt += chk_step


class GeneratorLen(object):
    def __init__(self, gen, length):
        """
        A class to wrap a generator, allowing it to have a length (which should be specified).
        Useful in situations where when we construct the generator, we know the length it's going to have,
        and would like the user of the generator to access this information the usual way.
        :param gen: the generator
        :param length: the length we want this generator to return when asked for it
        """
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def indexed_sliding_window_chunk_iter(it, chk_size, chk_step=None,
                                      start_at=None, stop_at=None,
                                      key=None, return_tail=True):
    """
      a function to get (an iterator of) segments (bt, tt) of chunks from (an iterator of) ordered timestamps,
      given a chk_size, chk_step, and a start_at time.
      :param it:
      :param chk_size:
      :param chk_step:
      :param start_at:
      :param key:
      :param return_tail:
      :return:

      1) If stop_at is not None and return_tail is False:
         will return all chunks with maximum element (in it or otherwise) less or equal to stop_at.

      2) If stop_at is not None and return_tail is True:
         will return all chunks with minimum element (in it or otherwise) less or equal to stop_at.

      3) If stop_at is None and return_tail is False:
         will return all chunks with maximum element less or equal the largest term in it.
         In other words, stop_at defaults to the last element of it and the behavior is as 1)

      4) If stop_at is None and return_tail is True:
         will return all chunks with minimum element less or equal the largest term in it.
         In other words, stop_at defaults to the last element of it and the behavior is as 2)

      See  /Users/MacBook/Desktop/Otosense/sound_sketch/ca/ChunkIteratorNB.html for examples with pictures

      #typical example, last two chunks contains 13 twice since chk_step=2 and return_tail=True
      >>> chk_size=2
      >>> chk_step=1
      >>> start_at=0
      >>> stop_at= None
      >>> return_tail = True
      >>> it = iter([0,2,3,7,9,10,13])
      >>> A = indexed_sliding_window_chunk_iter(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[0], [2], [2, 3], [3], [], [], [7], [7], [9], [9, 10], [10], [], [13], [13]]

      #similar to previous but with a start_at non zero and return_tail=False. The later condition
      #implies that 13 is returned only once
      >>> chk_size=3
      >>> chk_step=2
      >>> start_at=1
      >>> stop_at= None
      >>> return_tail = False
      >>> it = iter([0,2,3,7,9,10,13])
      >>> A = indexed_sliding_window_chunk_iter(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3], [3], [7], [7, 9], [9, 10], [13]]

      #this time stop_at is set to 11, consequently no chunk will contain a higher value, even though 13
      #is "within reach" of the last chunk since 11+2=13. This demonstrate the choice of having the
      #trailing chunk to not contain any value higher than stop_at.
      >>> chk_size=2
      >>> chk_step=4
      >>> start_at=2
      >>> stop_at= 11
      >>> return_tail = True
      >>> it = iter([0,2,3,7,9,10,13])
      >>> A = indexed_sliding_window_chunk_iter(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3], [7], [10]]

      #rather typical example. Since stop_at=None, each element of it belongs to at least one chunk
      >>> chk_size=5
      >>> chk_step=4
      >>> start_at=0
      >>> stop_at= None
      >>> return_tail = True
      >>> it = iter([0,2,3,7,9,10,13])
      >>> A = indexed_sliding_window_chunk_iter(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[0, 2, 3], [7], [9, 10], [13]]

      #chk_step is higher than ch_size here, we miss all terms in it which are not multiple of 6 here.
      >>> chk_size=1
      >>> chk_step=6
      >>> start_at=0
      >>> stop_at= None
      >>> return_tail = True
      >>> it = iter([0,2,3,7,9,10,13])
      >>> A = indexed_sliding_window_chunk_iter(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[0], [], []]

      #since the stop_at value is an element of it, we end up having it in at least one chunk. Even with a larger
      #chk_step of 3 we would not have 13 in any chunk since it is bove the stop_at value
      >>> chk_size=4
      >>> chk_step=2
      >>> start_at=1
      >>> stop_at= 10
      >>> return_tail = True
      >>> it = iter([0,2,3,7,9,10,13])
      >>> A = indexed_sliding_window_chunk_iter(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3], [3], [7], [7, 9, 10], [9, 10]]
      """

    if chk_step is None:
        chk_step = chk_size

    for obj in [chk_size, chk_step]:
        assert isinstance(obj, int), 'chk_size and chk_step must be integers'
        assert obj > 0, 'chk_size and chk_step must be positive'

    if key is None:  # key fetches the timestamps of the can signal, if none are given
        key = lambda x: x  # the elements are assumed to be the timestamps
    if start_at is None:
        x = next(it)  # get the first element
        start_at = key(x)  # ... and get the key for it
        it = chain([x], it)  # put that first element back in the iterator

    # initialize chunk
    chk = list()

    # initialize bt and tt (bottom and top of sliding window)
    bt = start_at
    tt = bt + chk_size

    if stop_at is not None and return_tail is False:

        for x in it:
            k = key(x)

            if tt > stop_at:
                return

            if k < bt:
                continue  # skip the remainder of the loop code until we get an element >= bt

            while tt <= k <= stop_at:
                yield chk
                bt += chk_step
                tt += chk_step
                chk = [i for i in chk if bt <= key(i) < tt]

            if bt <= k < tt:  # simplest case, we just append to chk
                chk.append(x)

            if k >= tt and k > stop_at:
                while tt <= stop_at:
                    yield chk
                    bt += chk_step
                    tt += chk_step
                    chk = [i for i in chk if bt <= key(i) < tt]

    if stop_at is not None and return_tail is True:

        for x in it:
            k = key(x)

            if bt > stop_at:
                return

            if k < bt:
                continue  # skip the remainder of the loop code until we get an element >= bt

            while tt <= k <= stop_at + chk_size:
                yield chk
                bt += chk_step
                tt += chk_step
                chk = [i for i in chk if bt <= key(i) < tt]

            if bt <= k < tt:  # simplest case, we just append to chk
                chk.append(x)

            if k >= tt and k > stop_at + chk_size:
                while tt <= stop_at + chk_size:
                    yield chk
                    bt += chk_step
                    tt += chk_step
                    chk = [i for i in chk if bt <= key(i) < tt]

    if stop_at is None and return_tail is False:

        for x in it:
            k = key(x)

            if k < bt:
                continue  # skip the remainder of the loop code until we get an element >= bt

            while k >= tt:
                yield chk
                bt += chk_step
                tt += chk_step
                chk = [i for i in chk if bt <= key(i) < tt]

            if bt <= k < tt:  # simplest case, we just append to chk
                chk.append(x)

        yield chk

    if stop_at is None and return_tail is True:

        for x in it:
            k = key(x)

            if k < bt:
                continue  # skip the remainder of the loop code until we get an element >= bt

            while k >= tt:
                yield chk
                bt += chk_step
                tt += chk_step
                chk = [i for i in chk if bt <= key(i) < tt]

            if bt <= k < tt:  # simplest case, we just append to chk
                chk.append(x)

        while len(chk) > 0:
            yield chk
            bt += chk_step
            tt += chk_step
            chk = [i for i in chk if bt <= key(i) < tt]


def fast_chunker(it, chk_size, chk_step=None, start_at=0, stop_at=None, return_tail=False):
    """
      a function to get (an iterator of) segments (bt, tt) of chunks from the iterator of {1,2,3...end},
      given a chk_size, chk_step, and a start_at time.
      :param it: iterator xrange(start,stop,1)
      :param chk_size: length of the chunks
      :param chk_step: step between chunks
      :param start_at: value from the iterator at which we begin building the chunks (inclusive)
      :param stop_at: last value from the iterator included in the chunks
      :param return_tail: if set to false, only the chunks with max element less than stop_at are yielded
      if set to true, any chunks with min value no more than stop_at are returned but they contain values no more
      than stop_at
      :return: an iterator of the chunks

      1) If stop_at is not None and return_tail is False:
         will return all chunks with maximum element (in it or otherwise) less or equal to stop_at.

      2) If stop_at is not None and return_tail is True:
         will return all chunks with minimum element (in it or otherwise) less or equal to stop_at.

      3) If stop_at is None and return_tail is False:
         will return all chunks with maximum element less or equal the largest term in it.
         In other words, stop_at defaults to the last element of it and the behavior is as 1)

      4) If stop_at is None and return_tail is True:
         will return all chunks with minimum element less or equal the largest term in it.
         In other words, stop_at defaults to the last element of it and the behavior is as 2)

      See  /Users/MacBook/Desktop/Otosense/sound_sketch/ca/ChunkIteratorNB.html for examples with pictures

      >>> chk_size = 3
      >>> chk_step = 2
      >>> start_at = 1
      >>> stop_at = None
      >>> return_tail = True
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size=chk_size, start_at=start_at, chk_step=chk_step,stop_at=stop_at,return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11], [11, 12, 13], [13, 14, 15], [15, 16]]

      >>> chk_size=3
      >>> chk_step=2
      >>> start_at=1
      >>> stop_at= 14
      >>> return_tail = True
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11], [11, 12, 13], [13, 14]]

      >>> chk_size=3
      >>> chk_step=2
      >>> start_at=1
      >>> stop_at= 14
      >>> return_tail = False
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11], [11, 12, 13]]

      >>> chk_size=2
      >>> chk_step=3
      >>> start_at=2
      >>> stop_at= None
      >>> return_tail = True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3], [5, 6], [8, 9], [11, 12], [14, 15], [17]]

      >>> chk_size=2
      >>> chk_step=3
      >>> start_at=2
      >>> stop_at= None
      >>> return_tail = False
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3], [5, 6], [8, 9], [11, 12], [14, 15]]

      >>> chk_size=3
      >>> chk_step=2
      >>> start_at=2
      >>> stop_at= 14
      >>> return_tail = True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10], [10, 11, 12], [12, 13, 14], [14]]

      >>> chk_size=2
      >>> chk_step=3
      >>> start_at=None
      >>> stop_at=None
      >>> return_tail = True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2], [4, 5], [7, 8], [10, 11], [13, 14], [16, 17]]

      >>> chk_size=5
      >>> chk_step=5
      >>> start_at=0
      >>> stop_at=17
      >>> return_tail = True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17]]


      >>> chk_size=5
      >>> chk_step=5
      >>> start_at=None
      >>> stop_at=16
      >>> return_tail = True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16]]


      >>> chk_size=50
      >>> chk_step=5
      >>> start_at=None
      >>> stop_at=16
      >>> return_tail = False
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      []


      >>> chk_size=50
      >>> chk_step=5
      >>> start_at=None
      >>> stop_at=16
      >>> return_tail = True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [11, 12, 13, 14, 15, 16], [16]]

      >>> chk_size=2
      >>> chk_step=3
      >>> start_at=2
      >>> stop_at=None
      >>> return_tail = True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3], [5, 6], [8, 9], [11, 12], [14, 15], [17]]


      >>> chk_size=2
      >>> chk_step=1
      >>> start_at=2
      >>> stop_at=15
      >>> return_tail = False
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]]

      >>> chk_size=1
      >>> chk_step=1
      >>> start_at=2
      >>> stop_at=15
      >>> return_tail = False
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]

      >>> chk_size=3
      >>> chk_step=2
      >>> start_at=1
      >>> stop_at=None
      >>> return_tail=True
      >>> it = iter(xrange(1,18,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11], [11, 12, 13], [13, 14, 15], [15, 16, 17], [17]]

      >>> chk_size=3
      >>> chk_step=8
      >>> start_at=1
      >>> stop_at=None
      >>> return_tail=True
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[1, 2, 3], [9, 10, 11]]

      >>> chk_size=3
      >>> chk_step=8
      >>> start_at=1000
      >>> stop_at=None
      >>> return_tail=True
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      []

      >>> chk_size=3
      >>> chk_step=8
      >>> start_at=1000
      >>> stop_at=None
      >>> return_tail=True
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      []

      >>> chk_size=3
      >>> chk_step=8000
      >>> start_at=10
      >>> stop_at=None
      >>> return_tail=True
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[10, 11, 12]]

      >>> chk_size=300
      >>> chk_step=8
      >>> start_at=10
      >>> stop_at=None
      >>> return_tail=False
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      []

      >>> chk_size=300
      >>> chk_step=8
      >>> start_at=10
      >>> stop_at=None
      >>> return_tail=True
      >>> it = iter(xrange(1,17,1))
      >>> A = fast_chunker(it, chk_size, chk_step, start_at, stop_at, return_tail=return_tail)
      >>> print list(A)
      [[10, 11, 12, 13, 14, 15, 16]]

       """

    if chk_step is None:
        chk_step = chk_size

    if stop_at is not None:
        assert isinstance(stop_at, int), 'stop_at should be an integer'

    # we set stop_at to be infinity by default
    if stop_at is None:
        stop_at = inf

    # looking up the value of the first element
    x = next(it)

    # consuming it until we reach start_at
    if start_at is not None:
        while x < start_at:
            x = next(it)
        it = chain([x], it)

    # setting the start_at to the first element of the iterator by default
    if start_at is None:
        start_at = x
        it = chain([x], it)  # put that first element back in the iterator

    # checking a few things
    assert isinstance(chk_size, int) and chk_size > 0, 'chk_size should be a positive interger'
    assert isinstance(chk_step, int) and chk_step > 0, 'chk_step should be a positive integer'
    assert isinstance(start_at, int), 'start_at should be an integer'
    assert stop_at > start_at, 'stop_at should be larger than start_at'

    # initializing chk
    chk = []

    # in that case, nothing to return
    if stop_at - start_at < chk_size and not return_tail:
        return

    # in that case only tails are returned
    finish = False
    if stop_at - start_at < chk_size and return_tail:
        bt = start_at
        tt = min(bt + chk_size, stop_at + 1)
        elem = next(it)
        while elem < tt and finish is False:
            chk.append(elem)
            try:
                elem = next(it)
            except:
                finish = True

        # yielding the chk as build above along with its shifts, truncated above by stop_at
        # and below by the sliding bt's
        while bt <= stop_at:
            yield chk
            bt += chk_step
            chk = [i for i in chk if i >= bt]
        return

    # if stop_at is not given, the number of chks will be infinity (or until it is exhausted)
    if stop_at is inf:
        n_chunks = inf
    else:  # compute the number of chk to return in the general case
        n_chunks = int(floor((stop_at - start_at + 1 - chk_size) / chk_step) + 1)

    # main loop for chk_step < chk_size
    if chk_step < chk_size:
        # counter for the number of chk returned
        chunk_number = 1
        # find and return the first chk
        chk = list(islice(it, 0, chk_size, 1))
        if return_tail or (not return_tail and len(chk) == chk_size):
            yield chk
        # number of terms to add to chk after shifting the previous one by chk_step
        n_it_to_add = chk_step
        # find and return the first chk
        finish = False
        while chunk_number < n_chunks and finish is False:
            chunk_number += 1
            chk = chk[chk_step:]
            for i in range(n_it_to_add):
                try:
                    to_add = next(it)
                    if to_add <= stop_at:
                        chk.append(to_add)
                except:
                    finish = True
                    break
            if len(chk) == chk_size or (return_tail is True and len(chk) > 0):
                yield chk

        # return the tail up to stop_at
        if return_tail and stop_at != inf:
            # find the current bt and add the chk_step to get the next bt
            bt = min(chk) + chk_step
            while bt <= stop_at:
                # anything from current bt to stop_at (included, thus the +1)
                chk = list(range(bt, stop_at + 1, 1))
                yield chk
                bt += chk_step

    # main loop for chk_step >= chk_size
    if chk_step >= chk_size:
        # number of terms to skip when going from one chk to the next
        n_it_to_skip = chk_step - chk_size
        # counter for the number of chk returned
        chunk_number = 1
        # find and return the first chk
        chk = list(islice(it, 0, chk_size, 1))
        yield chk
        # find and return the other ones
        finish = False
        while chunk_number < n_chunks and finish is False:
            chk = []
            for i in range(n_it_to_skip):
                next(it)
            for i in range(chk_size):
                try:
                    to_add = next(it)
                    chk.append(to_add)
                except:
                    finish = True
                    break
            if len(chk) == chk_size or (return_tail is True and len(chk) > 0):
                yield chk
            chunk_number += 1

        if return_tail and stop_at != inf:
            # if the min element of the last chk plus the chk_step is more than stop_at
            # there is nothing to return
            finish = False
            if min(chk) + chk_step > stop_at:
                return
            # otherwise there is exactly one more chk to yield
            chk = []
            for i in range(n_it_to_skip):
                next(it)
            elem = next(it)
            while elem <= stop_at and finish is False:
                try:
                    chk.append(elem)
                    elem = next(it)
                except:
                    finish = True
                    break
            yield chk


def first_elements_and_full_iter(it, n=1):
    """
    Given an iterator it, returns the pair (first_elements, it) (where it is the full original
    iterator).
    This is useful when you need to peek into an iterator before actually processing it (say
    because the way you will process it will depend on what you see in there).
    :param it: an iterator
    :param n: the number of first elements you want to peek at
    :return:
        first_elements: A list of the first n elements of the iterator
        it: The original (full) iterator
    """
    first_elements = take(n, it)
    return first_elements, itertools.chain(first_elements, it)


def batch(iterable, n=1, return_tail=True):
    """
    Iterator yielding batches of size n of the input iterable.
    See also grouper and seq_batch.
    :param iterable: in put iterable
    :param n: batch size
    :param return_tail: whether to return the last chunk (even if it's length is not the batch size)
    :return: an iterator
    """
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if return_tail and current_batch:
        yield current_batch


def grouper(iterable, n=1, fillvalue='drop'):
    """
    Returns an iterable that feeds tuples of size n corresponding to chunks of the input iterable.
    See also batch and seq_batch.
    :param iterable: Input iterable
    :param n: chunk (batch) size
    :param fillvalue: The element to use to fill the last chunk, or 'drop' to keep only elements of the iterable,
    meaning that the last tuple grouper will feed will be of size < n
    :return: An iterable that feeds you chunks of size n of the input iterable

    >>> list(grouper('ABCDEFG', 3, 'x'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    >>> list(grouper('ABCDEFG', 3, 'drop'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    """
    args = [iter(iterable)] * n
    if fillvalue == 'drop':
        return map(lambda x: [xx for xx in x if xx is not None], zip_longest(fillvalue=None, *args))
    else:
        return zip_longest(fillvalue=fillvalue, *args)


def seq_batch(seq, n=1, return_tail=True, fillvalue=None):
    """
    An iterator of equal sized batches of a sequence.
    See also grouper and seq_batch.
    :param seq: a sequence (should have a .__len__ and a .__getitem__ method)
    :param n: batch size
    :param return_tail:
        * True (default): Return the tail (what's remaining if the seq len is not a multiple of the batch size),
            as is (so the last batch might not be of size n
        * None: Return the tail, but fill it will the value specified in the fillvalue argument, to make it size n
        * False: Don't return the tail at all
    :param fillvalue: Value to be used to fill the tail if return_tail == None

    >>> seq = [1, 2, 3, 4, 5, 6, 7]
    >>> list(seq_batch(seq, 3, False))
    [[1, 2, 3], [4, 5, 6]]
    >>> list(seq_batch(seq, 3, True))
    [[1, 2, 3], [4, 5, 6], [7]]
    >>> list(seq_batch(seq, 3, None))
    [[1, 2, 3], [4, 5, 6], [7, None, None]]
    >>> list(seq_batch(seq, 3, None, 0))
    [[1, 2, 3], [4, 5, 6], [7, 0, 0]]
    """
    seq_len = len(seq)
    tail_len = seq_len % n
    for ndx in range(0, seq_len - tail_len, n):
        yield seq[ndx:(ndx + n)]

    # handing the tail...
    if tail_len > 0 and return_tail is not False:
        if return_tail is True:
            yield seq[(seq_len - tail_len):]
        else:
            t = list(seq[(seq_len - tail_len):]) + [fillvalue] * (n - tail_len)
            if isinstance(t, ndarray):
                yield array(t)
            else:
                try:
                    yield type(seq)(t)
                except Exception:
                    yield t


def sample_iterator(iterator, k, iterator_size=None):
    if iterator_size is None:
        iterator_size = len(iterator)
    for item in iterator:
        if random() < k / iterator_size:
            yield item
            k -= 1
        iterator_size -= 1
        if iterator_size == 0:
            raise StopIteration()


# def chunker(seq, size):
#     return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


# def chunker(seq, size, start=0):
#         for i in itertools.count(start, size):
#             yield seq[i: i + size]


def random_subset(iterator, K):
    """
    Uses reservoir sampling to get a sample from an iterator without knowing how many points there are
    in advance.
    """
    result = []

    for N, item in enumerate(iterator):
        if N <= K:
            result.append(item)
        else:
            s = int(random() * N)
            if s < K:
                result[s] = item

    return result


def print_iter_progress(iterator,
                        print_progress_every=None,
                        header_template="{hour:02.0f}:{minute:02.0f}:{second:02.0f} - iteration {iteration}",
                        data_msg_intro_str="",
                        data_to_string=None):
    """
    Wraps an iterator, allowing one to use the iterator as one would, but will print progress messages every
    print_progress_every iterations.

    header of print string can be specified through header_template
    data information can be printed too through data_msg_intro_str and data_to_string (a function) specifications

    Examples (but the doctest won't work, since time will be different):

    >>> for x in print_iter_progress(xrange(50), print_progress_every=10):
    ...     pass
    ...
    9:30:5 - iteration 0
    9:30:5 - iteration 10
    9:30:5 - iteration 20
    9:30:5 - iteration 30
    9:30:5 - iteration 40

    >>> for x in print_iter_progress(xrange(50),
    ...     print_progress_every=15,
    ...     data_msg_intro_str="data times two is: {data_str}",
    ...     data_to_string=lambda x: x * 2):
    ...     pass
    ...
    9:30:55 - iteration 0data times two is: 0
    9:30:55 - iteration 15data times two is: 30
    9:30:55 - iteration 30data times two is: 60
    9:30:55 - iteration 45data times two is: 90
    """
    if print_progress_every is None:
        for x in iterator:
            yield x
    else:
        print_template = header_template + data_msg_intro_str
        for i, x in enumerate(iterator):
            if mod(i, print_progress_every) == 0:
                t = datetime.now().time()
                if data_to_string is None:
                    print((print_template.format(hour=t.hour, minute=t.minute, second=t.second, iteration=i)))
                else:
                    print((print_template.format(hour=t.hour, minute=t.minute, second=t.second, iteration=i,
                                                 data_str=data_to_string(x))))
            yield x


def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def all_subsets_of(iterable, include_empty_set=True):
    if include_empty_set is True:
        start = 0
    else:
        start = 1
    n = len(list(iterable))
    return chain(*map(lambda x: combinations(iterable, x), range(start, n + 1)))


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def tabulate(function, start=0):
    "Return function(0), function(1), ..."
    return map(function, itertools.count(start))


def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        itertools.collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def quantify(iterable, pred=bool):
    "Count how many times the predicate is true"
    return sum(map(pred, iterable))


def padnone(iterable):
    """Returns the sequence elements and then returns None indefinitely.

    Useful for emulating the behavior of the built-in map() function.
    """
    return chain(iterable, itertools.repeat(None))


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return chain.from_iterable(itertools.repeat(tuple(iterable), n))


def dotproduct(vec1, vec2):
    return sum(map(itertools.operator.mul, vec1, vec2))


def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)


def repeatfunc(func, times=None, *args):
    """Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return itertools.starmap(func, itertools.repeat(args))
    return itertools.starmap(func, itertools.repeat(args, times))


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper_no_fill(iterable, n):  # untested
    "grouper_no_fill('ABCDEFG', 3) --> ABC DEF G"
    args = [iter(iterable)] * n
    return map(lambda x: [_f for _f in x if _f], zip_longest(fillvalue=None, *args))


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.slice(nexts, pending))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(iterable, key=None):
    """
    List unique elements, preserving order. Remember all elements ever seen.
    >>> list(unique_everseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']
    >>> import string
    >>> list(unique_everseen('ABBCcAD', string.lower))
    ['A', 'B', 'C', 'D']
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable, key=None):
    """
    List unique elements, preserving order. Remember only the element just seen.
    >>> list(unique_justseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> import string
    >>> list(unique_justseen('ABBCcAD', string.lower))
    ['A', 'B', 'C', 'A', 'D']
    """
    return map(next, map(itemgetter(1), itertools.groupby(iterable, key)))


def iter_except(func, exception, first=None):
    """ Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like __builtin__.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.

    Examples:
        bsddbiter = iter_except(db.next, bsddb.error, db.first)
        heapiter = iter_except(functools.partial(heappop, h), IndexError)
        dictiter = iter_except(d.popitem, KeyError)
        dequeiter = iter_except(d.popleft, IndexError)
        queueiter = iter_except(q.get_nowait, Queue.Empty)
        setiter = iter_except(s.pop, KeyError)

    """
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass


def random_product(*args, **kwds):
    "Random selection from itertools.product(*args, **kwds)"
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    return tuple(itertools.random.choice(pool) for pool in pools)


def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(itertools.random.sample(pool, r))


def random_combination(iterable, r):
    "Random selection from combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(itertools.random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable, r):
    "Random selection from combinations_with_replacement(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(itertools.random.randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)


def tee_lookahead(t, i):
    """Inspect the i-th upcomping value from a tee object
       while leaving the tee object at its current position.

       Raise an IndexError if the underlying iterator doesn't
       have enough values.

    """
    for value in islice(t.__copy__(), i, None):
        return value
    raise IndexError(i)


def chunker(it, chk_size, chk_step=None, start_at=0, stop_at=None, return_tail=False):
    # TODO: Handle true iterator case
    if chk_step is None:
        chk_step = chk_size

    if hasattr(it, '__getslice__'):
        if stop_at is None:
            stop_at = len(it)
        else:
            stop_at = min(len(it), stop_at)
        if not return_tail:
            # print chk_size, start_at, stop_at
            stop_at = start_at + chk_size * ((stop_at - start_at) // chk_size)
            # print stop_at
        it = it[start_at:stop_at]
        bt = 0
        tt = bt + chk_size
        max_bt = stop_at - start_at
        while bt < max_bt:
            yield it[bt:tt]
            bt += chk_step
            tt += chk_step
    else:
        if chk_size <= chk_step:
            if start_at > 0:
                for i in range(start_at):
                    _ = next(it)  # throw away the first items
            if stop_at is not None:
                stop_at -= start_at
                max_bt = stop_at - chk_size
                n_chks_float = max_bt / float(chk_step)
                n_chks = int(n_chks_float)
                if n_chks != n_chks_float:
                    if return_tail:
                        n_chks += 1
            else:
                n_chks = inf

            chk_idx = 0
            while chk_idx < n_chks:
                x = list(islice(it, chk_size))
                if len(x) < chk_size:  # TODO: Might be able to handle this case outside the loop?
                    if return_tail:
                        yield x
                    break
                yield x
                chk_idx += 1
        else:
            from warnings import warn
            warn("Haven't implemented the true iterator case")
            for x in chunker(list(it), chk_size, chk_step=chk_step, start_at=start_at, stop_at=stop_at,
                             return_tail=return_tail):
                yield x
