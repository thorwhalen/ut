from functools import reduce, partial
from collections import defaultdict
import inspect
import re
from warnings import warn
import pandas as pd

super_methods_p = re.compile('super\(\)\.(?P<super_method>\w+)\(')

ordered_unik_elements = partial(reduce,
                                lambda unik, new_items: unik + [x for x in new_items if x not in unik])


class MethodNotFoundInMro(NotImplementedError):
    pass


def find_super_methods_in_code_line(code_line):
    r = super_methods_p.search(code_line)
    if r:
        return r.groupdict().get('super_method', None)


def find_super_methods_in_code_object(code_obj):
    code_lines, _ = inspect.getsourcelines(code_obj)
    super_methods = filter(None, map(find_super_methods_in_code_line,
                                     filter(lambda x: not x.startswith('#'),
                                            map(str.strip, code_lines))))
    return list(super_methods)


def find_super_methods_in_func(func):
    if callable(func) and hasattr(func, '__code__'):
        return find_super_methods_in_code_object(func.__code__)
    else:
        return []


def method_calls_super_method_of_same_name(method):
    super_methods = find_super_methods_in_func(method)
    return method.__name__ in super_methods


def mro_for_method(cls, method, mro=None, method_not_found_error=True, include_overridden_methods=False):
    if not isinstance(method, str):
        method = method.__name__
    if mro is None:
        mro = cls.mro()
    mro_length = len(mro)
    super_methods = []
    _super_method_expected = False
    super_cls = None
    for i, super_cls in enumerate(mro, 1):
        method_func = super_cls.__dict__.get(method, None)
        if method_func is not None:  # we found the class where the method that will be called is
            super_methods.append(super_cls)  # add this to the list (it's the first element)
            # get the list of super methods called in the code:
            if not method_calls_super_method_of_same_name(method_func):
                _super_method_expected = False
                if not include_overridden_methods:
                    break
            else:
                _super_method_expected = True  # flag to indicate that a super method is expected
                # if our target method is called (with super) in that code...
                # extend our list with the the list of super methods called further in the mro...
                if i >= mro_length:
                    raise ValueError("There was a super method call, but no more classes in the mro!")

    if super_cls is not None and _super_method_expected:
        warn("Method {} defined in {}, but call to super method has no resolution.".format(method, super_cls))

    if method_not_found_error and len(super_methods) == 0:
        raise MethodNotFoundInMro("Method {} isn't implemented in the mro of class {}".format(method, cls))

    return super_methods


def _no_dunder(x):
    return not (x.startswith('__') and x.endswith('__'))


def mk_cls_identifier(cls_identifier='name'):
    if cls_identifier == 'module_path':
        def id_of_cls(cls):
            return cls.__module__ + '.' + cls.__qualname__
    elif cls_identifier == 'name':
        def id_of_cls(cls):
            return cls.__qualname__
    else:
        def id_of_cls(cls):
            return cls
    return id_of_cls


def method_resolutions(cls, methods=None, cls_identifier=None, method_not_found_error=True,
                       include_overridden_methods=False):
    """

    :param cls:
    :param methods:
    :param cls_identifier:
    :param method_not_found_error:
    :return:
    >>> class A:
    ...     def foo(self):
    ...         return 42
    ...     def hello(self):
    ...         pass
    >>> class B(A):
    ...     def hello(self):
    ...         super().hello()
    >>> class C(A):
    ...     def foo(self):
    ...         super().foo()
    ...     def bar(self):
    ...         super().bar()  # if A is the next in line in the mro, this should fail, since A has no bar
    ...     def hello(self):
    ...         super().hello()
    >>> class BC(B, C):
    ...     def foo(self):
    ...         print('hello BC')
    ...         super().bar()  # a call to bar within foo. Should be ignored by the search for super().foo()
    ...         super().foo()
    ...     def hello(self):
    ...         super().hello()
    >>> class CB(C, B):
    ...     def foo(self):
    ...         # super().foo(), this comment is there just to test that the super().foo() it actually ignored
    ...         pass
    ...     def hello(self):
    ...         super().hello()
    >>> for cls in [A, B, C, BC, CB]:
    ...     print("---- {} mro ----".format(cls.__name__))
    ...     print(", ".join(map(lambda x: x.__name__, cls.__mro__)))
    ---- A mro ----
    A, object
    ---- B mro ----
    B, A, object
    ---- C mro ----
    C, A, object
    ---- BC mro ----
    BC, B, C, A, object
    ---- CB mro ----
    CB, C, B, A, object
    >>>
    >>> method_resolutions(BC, ['__init__', 'hello', 'foo'], cls_identifier='name')
    {'__init__': ['object'], 'hello': ['BC', 'B', 'C', 'A'], 'foo': ['BC', 'C', 'A']}
    >>> method_resolutions(CB, ['__init__', 'hello', 'foo'], cls_identifier='name')
    {'__init__': ['object'], 'hello': ['CB', 'C', 'B', 'A'], 'foo': ['CB']}
    >>>
    >>> import warnings; warnings.filterwarnings('error')
    >>> try:
    ...     res = method_resolutions(CB, ['bar'], cls_identifier='name')
    ... except UserWarning:
    ...     print("Expected this UserWarning: C.bar calls super().bar but there's not bar further in mro")
    Expected this UserWarning: C.bar calls super().bar but there's not bar further in mro
    """
    if methods is None:
        methods = ['__init__'] + list(filter(_no_dunder, cls.__dict__.keys()))
    id_of_cls = mk_cls_identifier(cls_identifier)
    _mro_for_method = {
        method: list(map(id_of_cls,
                         mro_for_method(cls, method, None, method_not_found_error, include_overridden_methods)))
        for method in methods}

    return _mro_for_method


def df_of_method_resolutions(cls, methods=None, cls_identifier='name', method_not_found_error=True,
                             include_overridden_methods=False):
    _mro_for_method = method_resolutions(
        cls, methods, cls_identifier, method_not_found_error, include_overridden_methods)
    d = {method: {k: i for i, k in enumerate(resolutions)} for method, resolutions in _mro_for_method.items()}
    d = pd.DataFrame(d).T
    methods = list(_mro_for_method.keys())
    id_of_cls = mk_cls_identifier(cls_identifier)
    d_cols = set(d.columns)
    cols = [c for c in map(id_of_cls, cls.mro()) if c in d_cols]
    d = d.loc[methods, cols].fillna(-1).astype(int)
    d[d == -1] = ""
    return d


def assert_cls(cls):
    if isinstance(cls, type):
        return cls
    else:
        return cls.__class__  # it's probably an instance, and we want the class


def class_path_str(cls):
    return assert_cls(cls).__module__ + '.' + cls.__name__


def mro_str_with_indents(cls, indent=4):
    cls = assert_cls(cls)
    _indent_of_cls = dict()

    def indent_of_cls(_cls):
        for __cls in reversed(list(_indent_of_cls.keys())):
            if _cls in __cls.__bases__:
                _indent_of_cls[_cls] = _indent_of_cls[__cls] + indent
                return _indent_of_cls[_cls]
        # if got so far...
        _indent_of_cls[_cls] = 0
        return _indent_of_cls[_cls]

    s = ''
    for _cls in cls.mro():
        _indent = indent_of_cls(_cls)
        s += ' ' * _indent + class_path_str(_cls) + '\n'
    return s


def print_mro(cls, indent=4):
    print(mro_str_with_indents(cls, indent))


def all_methods_in_the_order_they_were_encountered_in_mro(cls):
    all_methods = []
    for _cls in cls.mro():
        methods = [k for k, v in _cls.__dict__.items() if callable(v)]
        all_methods = all_methods + [method for method in methods if method not in all_methods]
    return all_methods


def mro_class_methods(cls, include=None, exclude=None, cls_identifier='name', include_overridden_methods=False):
    """
    Get the methods that each class of the mro contains, organized by class.
    :param cls: class to analyze
    :param include: method (names) to include
    :param exclude: method (names) to exclude
    :param cls_identifier: how to represent classes (the class object itself, the module path, just the name (default))
    :param include_overridden_methods: If False (default), will not include methods that have been overridden
    :return: A {class: methods, ...} dict

    >>> class A:
    ...     def foo(self):
    ...         return 42
    ...     def hello(self):
    ...         pass
    >>> class B(A):
    ...     def hello(self):
    ...         super().hello()
    >>> class C(A):
    ...     def foo(self):
    ...         super().foo()
    ...     def bar(self):
    ...         super().bar()  # if A is the next in line in the mro, this should fail, since A has no bar
    ...     def hello(self):
    ...         super().hello()
    >>> class BC(B, C):
    ...     def foo(self):
    ...         print('hello BC')
    ...         super().bar()  # a call to bar within foo. Should be ignored by the search for super().foo()
    ...         super().foo()
    ...     def hello(self):
    ...         super().hello()
    >>> class CB(C, B):
    ...     def foo(self):
    ...         # super().foo(), this comment is there just to test that the super().foo() it actually ignored
    ...         pass
    ...     def hello(self):
    ...         super().hello()
    >>> mro_class_methods(CB)
    {'CB': ['foo', 'hello'], 'C': ['hello', 'bar'], 'B': ['hello'], 'A': ['hello'], 'object': ['__init__']}
    >>> mro_class_methods(BC)
    {'BC': ['foo', 'hello'], 'C': ['foo', 'hello', 'bar'], 'A': ['foo', 'hello'], 'B': ['hello'], 'object': ['__init__']}
    """
    all_methods = all_methods_in_the_order_they_were_encountered_in_mro(cls)

    # resolve inclusion/exclusion
    if include:
        all_methods = [method for method in all_methods if method in include]
    else:
        if not exclude:
            # if not include or exclude, exclude all base object methods except __init__
            exclude = set([k for k in object.__dict__.keys() if k not in {'__init__'}])
        all_methods = [method for method in all_methods if method not in exclude]

    # handle methods calling super methods of the same name (and warn if supers without resolution)
    cls_resolution_for_method = {
        method: mro_for_method(cls, method, include_overridden_methods=include_overridden_methods)
        for method in all_methods
    }

    methods_of_cls = defaultdict(list)
    for method, _classes in cls_resolution_for_method.items():
        for _cls in _classes:
            methods_of_cls[_cls].append(method)
    id_of_cls = mk_cls_identifier(cls_identifier)
    return {id_of_cls(_cls): methods for _cls, methods in methods_of_cls.items()}


def df_of_mro_class_methods(cls, include=None, exclude=None, cls_identifier='name', include_overridden_methods=False):
    methods_of_cls = mro_class_methods(cls, include, exclude, cls_identifier)
    df = pd.DataFrame(index=ordered_unik_elements(methods_of_cls.values()), columns=methods_of_cls.keys())
    df = df.fillna('')
    for cls_name, methods in methods_of_cls.items():
        for method in methods:
            df.loc[method, cls_name] = method
    return df
