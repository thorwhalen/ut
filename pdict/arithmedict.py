import operator


def _apply_op(op, d1, dflt_1, d2, dflt_2):
    if isinstance(d2, dict):
        out = dict()
        for k, v1 in d1.items():
            v2 = d2.get(k, dflt_2)
            out[k] = op(v1, v2)
        for k in d2:  # take care of the remainder (those keys in dict_2 that were not in dict_1)
            if k not in out:
                out[k] = op(dflt_1, d2[k])
    else:
        out = {k: op(v, d2) for k, v in d1.items()}
    return out


def _mk_op_method(op, dflt_1, dflt_2, for_reflexive_op=False):
    if not for_reflexive_op:
        def op_method(self, d):
            return self.__class__(_apply_op(op, self, dflt_1, d, dflt_2))
    else:
        def op_method(self, d):
            return self.__class__(_apply_op(op, d, dflt_1, self, dflt_2))

    return op_method


def _mk_unary_op_method(op):
    def op_method(self):
        return self.__class__({k: op(v) for k, v in self.items()})

    return op_method


# Notes: Not sure if mod has an identity.
# __and__ would have one theoretically (infinity 1s). didn't think about xor.
# __matmul__ has an identity DEPENDING ON square and if so, PER dimensions of matrix
# TODO: not sure of the appropriate defaults for __pow__.
_ops_and_identity = [
    ({'__add__', '__sub__', '__lshift__', '__rshift__', '__or__'}, 0),
    ({'__mul__', '__truediv__', '__floordiv__', '__pow__'}, 1),
    ({'__mod__', '__and__', '__xor__', '__matmul__'}, None)
]

_unary_ops = {'__pos__', '__neg__', '__abs__', '__invert__'}


class ArithmeDict(dict):
    """A dict, with arithmetic.
    A unary operator is just applied to all values.
    When a dict operates with a number, the operation is applied to each value of the dict.
    When a dict operates with another dict, the keys are aligned and the operation applied to the aligned values.

    The class is meant to be used in situations where pandas.Series would be used to operate with (sparse) vectors
    such as word counts, etc.

    Performance:

    In a nutshell, if you use pandas already in your app, then use pandas.Series instead.
    But, if you want weight packages (pandas isn't light), or have small dicts you want to operate on, use ArithmeDict.

    Note that both construction and operation are faster on ArithmeDict, for smaller sets.

    ```
    import pandas as pd

    t = ArithmeDict(a=1, b=2)
    tt = ArithmeDict(b=3, c=4)
    %timeit t + tt
    # 1.41 µs ± 41.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    ### versus ###

    t = pd.Series(dict(a=1, b=2))
    tt = pd.Series(dict(b=3, c=4))
    %timeit t + tt  # and not even what we want (see later)
    # 405 µs ± 7.65 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    % timeit pd.Series.add(t, tt, fill_value=0).to_dict()
    # 410 µs ± 11.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    ### but ###
    t = ArithmeDict({i: i for i in range(10000)})
    tt = ArithmeDict({i: i for i in range(5000, 15000)})
    %timeit t + tt
    # 3.22 ms ± 98.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    ### not so far from ###
    t = pd.Series({i: i for i in range(10000)})
    tt = pd.Series({i: i for i in range(5000, 15000)})
    %timeit pd.Series.add(t, tt, fill_value=0).to_dict()
    3.71 ms ± 100 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # and actually much slower than:
    %timeit pd.Series.add(t, tt, fill_value=0)
    575 µs ± 17.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    ```

    On the other hand, memory usage is inconclusive, because I don't know how to actually make the comparison.
    ```
    import pickle, sys, pandas

    t = ArithmeDict({i: i for i in range(10000)})
    sys.getsizeof(t), len(pickle.dumps(t))
    # (295032, 59539)

    t = pandas.Series({i: i for i in range(10000)})
    sys.getsizeof(t), len(pickle.dumps(t))
    # (160032, 240666)
    ```


    Notes for enhancement:

    When dict operates with/on a dict, and therefore we need to align keys,
    there are different merge and reduce options that may or may not make sense according to the value type and context.
    For example, should we really keep all keys and use operand defaults to get their values, or just drop
    those fields all together?
    Also, if we choose to keep all keys, what should the operand default be.
    Sometimes it might depend on the other operand (example matmul), or need to be created (example __concat__,
    since don't want the mutable list as a default), etc.

    >>> d1 = ArithmeDict(a=1, b=2)
    >>> d2 = ArithmeDict(b=3, c=4)
    >>>
    >>> # These are still dicts
    >>> isinstance(d1, dict)
    True
    >>> # and display as such
    >>> d1
    {'a': 1, 'b': 2}
    >>> d2
    {'b': 3, 'c': 4}
    >>>
    >>> # Unary operators (just applied to all values)
    >>> assert -d1 == {'a': -1, 'b': -2}
    >>> assert abs(-d1) == d1  # ... and in case that doesn't look impressive enough..
    >>> assert abs(ArithmeDict(a=-1, b=2, c=-3)) == {'a': 1, 'b': 2, 'c': 3}
    >>>
    >>> # An operation with a number is transferred to the values of the dict (applied to each).
    >>> assert d1 + 10 == {'a': 11, 'b': 12}
    >>> assert d1 - 10 == {'a': -9, 'b': -8}
    >>> assert d1 * 10 == {'a': 10, 'b': 20}
    >>> assert d1 / 10 == {'a': 0.1, 'b': 0.2}
    >>> assert d1 // 2 == {'a': 0, 'b': 1}
    >>> assert d1 ** 2 == {'a': 1, 'b': 4}
    >>> assert d2 % 2 == {'b': 1, 'c': 0}
    >>> assert d2 % 3 == {'b': 0, 'c': 1}
    >>> assert d2 >> 1 == {'b': 1, 'c': 2}  # shift all bits by one bit to the right
    >>> assert d2 << 1 == {'b': 6, 'c': 8}  # shift all bits by one bit to the left
    >>>
    >>> # An operation with another dict will align the keys and apply the operation to the aligned values.
    >>> assert d1 + d2 == {'a': 1, 'b': 5, 'c': 4}
    >>> assert d1 - d2 == {'a': 1, 'b': -1, 'c': -4}
    >>> assert d1 * d2 == {'a': 1, 'b': 6, 'c': 4}
    >>> assert d1 / d2 == {'a': 1, 'b': 0.6666666666666666, 'c': 0.25}
    >>> assert d2 // d1 == {'b': 1, 'c': 4, 'a': 1}
    >>> assert d1 ** d2 == {'a': 1, 'b': 8, 'c': 1}
    >>> assert ArithmeDict(a=10, b=10) % dict(a=3, b=4) == {'a': 1, 'b': 2}
    >>> assert d1 << d2 == {'a': 1, 'b': 16, 'c': 0}  # shifting bits
    >>> assert d1 + {'b': 3, 'c': 4} == {'a': 1, 'b': 5, 'c': 4}  # works when the right side is a normal dict
    >>> assert d1 + ArithmeDict() == d1
    >>> assert ArithmeDict() - d1 == -d1
    """

    for op in _unary_ops:
        locals()[op] = _mk_unary_op_method(getattr(operator, op))

    for ops, identity_val in _ops_and_identity:
        for op in ops:
            op_func = getattr(operator, op)
            locals()[op] = _mk_op_method(op_func, dflt_1=identity_val, dflt_2=identity_val, for_reflexive_op=False)
