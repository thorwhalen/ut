from collections import namedtuple
from functools import wraps


def namedtuple_to_dict(nt):
    """
    >>> from collections import namedtuple
    >>> NT = namedtuple('MyTuple', ('foo', 'hello'))
    >>> nt = NT(1, 42)
    >>> nt
    MyTuple(foo=1, hello=42)
    >>> d = namedtuple_to_dict(nt)
    >>> d
    {'foo': 1, 'hello': 42}
    """
    return {field: getattr(nt, field) for field in nt._fields}


def dict_to_namedtuple(d, namedtuple_obj=None):
    """
    >>> from collections import namedtuple
    >>> NT = namedtuple('MyTuple', ('foo', 'hello'))
    >>> nt = NT(1, 42)
    >>> nt
    MyTuple(foo=1, hello=42)
    >>> d = namedtuple_to_dict(nt)
    >>> d
    {'foo': 1, 'hello': 42}
    >>> dict_to_namedtuple(d)
    NamedTupleFromDict(foo=1, hello=42)
    >>> dict_to_namedtuple(d, nt)
    MyTuple(foo=1, hello=42)
    """
    if namedtuple_obj is None:
        namedtuple_obj = 'NamedTupleFromDict'
    if isinstance(namedtuple_obj, str):
        namedtuple_name = namedtuple_obj
        namedtuple_cls = namedtuple(namedtuple_name, tuple(d.keys()))
    elif isinstance(namedtuple_obj, tuple) and hasattr(namedtuple_obj, '_fields'):
        namedtuple_cls = namedtuple_obj.__class__
    elif isinstance(namedtuple_obj, type):
        namedtuple_cls = namedtuple_obj
    else:
        raise TypeError(f"Can't resolve the nametuple class specification: {namedtuple_obj}")

    return namedtuple_cls(**d)


def update_fields_of_namedtuple(nt: tuple, *, name_of_output_type=None, remove_fields=(), **kwargs):
    """Replace fields of namedtuple
    >>> from collections import namedtuple
    >>> NT = namedtuple('NT', ('a', 'b', 'c'))
    >>> nt = NT(1,2,3)
    >>> nt
    NT(a=1, b=2, c=3)
    >>> update_fields_of_namedtuple(nt, c=3000)  # replacing a single field
    NT(a=1, b=2, c=3000)
    >>> update_fields_of_namedtuple(nt, c=3000, a=1000)  # replacing two fields
    NT(a=1000, b=2, c=3000)
    >>> update_fields_of_namedtuple(nt, a=1000, c=3000)  # see that the original order doesn't change
    NT(a=1000, b=2, c=3000)
    >>> update_fields_of_namedtuple(nt, b=2000, d='hello')  # replacing one field and adding a new one
    UpdatedNT(a=1, b=2000, c=3, d='hello')
    >>> # Now let's try controlling the name of the output type, remove fields, and add new ones
    >>> update_fields_of_namedtuple(nt, name_of_output_type='NewGuy', remove_fields=('a', 'c'), hello='world')
    NewGuy(b=2, hello='world')
    """

    output_type_can_be_the_same_as_input_type = (not remove_fields) and set(kwargs.keys()).issubset(nt._fields)
    d = dict(namedtuple_to_dict(nt), **kwargs)
    for f in remove_fields:
        d.pop(f)

    if output_type_can_be_the_same_as_input_type and name_of_output_type is None:
        return dict_to_namedtuple(d, nt.__class__)
    else:
        name_of_output_type = name_of_output_type or f'Updated{nt.__class__.__name__}'
        return dict_to_namedtuple(d, name_of_output_type)


def sub_namedtuple(nt: tuple, index):
    """

    Args:
        nt:
        index: index: a (python identifier) string, a tuple thereof, or a (numerical) slice

    Returns:

    >>> from collections import namedtuple
    >>> T = namedtuple('blah', ('foo', 'bar', 'pi'), defaults=(0, 3.14))
    >>> t = T(2, 10)
    >>> t
    blah(foo=2, bar=10, pi=3.14)
    >>> sub_namedtuple(t, 'foo')
    2
    >>> sub_namedtuple(t, ('foo', 'pi'))
    blah(foo=2, pi=3.14)
    >>> sub_namedtuple(t, slice(None, 2, None))  # t[:2]
    blah(foo=2, bar=10)
    >>> sub_namedtuple(t, slice(-2, None, None))  # t[-2:]
    blah(bar=10, pi=3.14)
    """
    # `type(self)` can result in issues in case of multiple inheritance.
    # But shouldn't be an issue here.
    if isinstance(index, str):
        return getattr(nt, index)
    else:
        if isinstance(index, tuple):
            if hasattr(tuple, '_fields'):
                index_field_names = index._fields
            else:
                index_field_names = index
            index_field_values = [getattr(nt, x) for x in index_field_names]  # TODO: Potential bisect optimization
            cls = namedtuple(nt.__class__.__name__, index_field_names)
            return cls(*index_field_values)
        elif isinstance(index, slice):
            cls = namedtuple(nt.__class__.__name__, nt._fields[index])
            return cls(*nt[index])
        else:
            return nt[index]
            # raise TypeError("I can only handle str, tuple, slice")


def is_sub_list(sub, lst):
    indices = list(map(lst.index, sub))
    return indices == sorted(indices)


def is_sub_namedtuple(t, tt, name_equality=False, order_equality=True):
    """Says if t is a sub-namedtuple of tt or not.
    For t to be a sub-namedtuple of tt, the following need to apply:
        * t._fields must be a subset of tt._fields
        * The values for these t._fields must be the same as those in tt._fields
        * if name_equality == True, we must also have t.__class__.__name__ == tt.__class__.__name__
        * if order_equality == False, we must also have the fields in the same order

    >>> t = dict_to_namedtuple(dict(a=0, c=2))
    >>> tt = dict_to_namedtuple(dict(a=0, d=3, c=2))
    >>> ttt = dict_to_namedtuple(dict(a=0, b=1, c=2, d=3))
    >>> t2 = dict_to_namedtuple(dict(a=0, c=3))
    >>>
    >>> is_sub_namedtuple(t, tt)
    True
    >>> is_sub_namedtuple(t, ttt)
    True
    >>> is_sub_namedtuple(tt, ttt)
    False
    >>> is_sub_namedtuple(tt, ttt, order_equality=False)
    True
    >>> is_sub_namedtuple(tt, t)
    False
    >>> is_sub_namedtuple(t, t2)
    False
    >>> from collections import namedtuple
    >>> is_sub_namedtuple(t, namedtuple('AnotherName', ('a', 'c'))(*t))
    True
    >>> is_sub_namedtuple(t, namedtuple('AnotherName', ('a', 'c'))(*t), name_equality=True)
    False
    """
    return set(t._fields).issubset(tt._fields) \
           and all(getattr(t, f) == getattr(tt, f) for f in t._fields) \
           and ((not name_equality) or (t.__class__.__name__ == tt.__class__.__name__)) \
           and ((not order_equality) or is_sub_list(t, tt))


def print_is_sub_namedtuple_with_explanation(t, tt, name_equality=False, order_equality=True):
    """Says if t is a sub-namedtuple of tt or not.
    For t to be a sub-namedtuple of tt, the following need to apply:
        * t._fields must be a subset of tt._fields
        * The values for these t._fields must be the same as those in tt._fields
        * if name_equality == True, we must also have t.__class__.__name__ == tt.__class__.__name__
        * if order_equality == False, we must also have the fields in the same order
    """
    if not set(t._fields).issubset(tt._fields):
        print("t._fields not subset of tt._fields")
    if not all(getattr(t, f) == getattr(tt, f) for f in t._fields):
        print("some values of t are different in tt")
    if not ((not name_equality) or (t.__class__.__name__ == tt.__class__.__name__)):
        print("names are not the same")
    if not ((not order_equality) or is_sub_list(t, tt)):
        print("not the same order")


# TODO: Use sub_namedtuple to not repeat code
@wraps(namedtuple)
def nametuple_extended(typename, field_names, **kwargs):
    cls = namedtuple(typename, field_names, **kwargs)

    def getitem(self, index):
        # `type(self)` can result in issues in case of multiple inheritance.
        # But shouldn't be an issue here.
        if isinstance(index, str):
            return getattr(self, index)
        elif isinstance(index, tuple):
            index_field_names = index
            index_field_values = [getattr(self, x) for x in index_field_names]  # TODO: Potential bisect optimization
            cls = namedtuple(typename, index_field_names)
            cls.__getitem__ = getitem
            return cls(*index_field_values)

        value = super(type(self), self).__getitem__(index)
        if isinstance(index, slice):
            cls = namedtuple(typename, field_names[index])
            cls.__getitem__ = getitem
            value = cls(*value)
        return value

    #     def __add__()

    cls.__getitem__ = getitem
    return cls


def nametuple_extended_from_dict(d, typename='NamedTupleExtended'):
    return nametuple_extended(typename, field_names=list(d.keys()))(**d)


nametuple_extended.from_dict = nametuple_extended_from_dict

# T = nametuple_extended('blah', ('foo', 'bar', 'pi'), defaults=(0, 3.14))
# t = T(2, 10)
# t['foo', 'pi']
