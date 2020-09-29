from inspect import getsource, getsourcelines, signature
from typing import Any, Optional
import os

path_sep = os.path.sep


def not_dunder(a):
    return not a.startswith('__')


def aval_gen(o, filt=not_dunder):
    for a in filter(filt, dir(o)):
        yield a, getattr(o, a)


def show_attrs(o, filt=not_dunder):
    print(*(f"{a}:\t{v}" for a, v in aval_gen(o, filt)), sep='\n')


def print_source(o, start=None, end=None, doc=True):
    """Prints the source code of the input object. Can specify lines to print with start and end.
    >>> print_source(print_source, 0, 1)
    def print_source(o, start=None, end=None, doc=True):
    <BLANKLINE>
    >>> print_source(print_source, 11)
        _lines, _ = getsourcelines(o)
        print(''.join(_lines[slice(start, end)]))
    <BLANKLINE>
    """
    # TODO: Make doc=False work
    _lines, _ = getsourcelines(o)
    print(''.join(_lines[slice(start, end)]))


def print_signature(func, sep: Optional[str] = '\n', prefix: str = '', suffix: str = ''):
    """Print the signature of a callable
    :param func: Callable to print the signature of
    :param sep: If None, will print the signature as inspect.signature would.
        If a string, will use the string to separate the parameter specifications

    >>> print_signature(print_signature)
    func
    sep: Union[str, NoneType] = '\\n'
    prefix: str = ''
    suffix: str = ''
    >>> print_signature(print_signature, None)
    (func, sep: Union[str, NoneType] = '\\n', prefix: str = '', suffix: str = '')
    >>> print_signature(print_signature, '\\n * ', prefix=' * ', suffix='\\n')
     * func
     * sep: Union[str, NoneType] = '\\n'
     * prefix: str = ''
     * suffix: str = ''
    <BLANKLINE>
    """
    sig = signature(func)
    if sep is None:
        print(prefix + str(sig) + suffix)
    else:
        print(prefix + sep.join(map(str, sig.parameters.values())) + suffix)


def submodule_path_strings(module):
    from py2store.filesys import FileCollection

    root_name = os.path.dirname(module.__file__)
    root_name_length = len(root_name) + 1
    for pyfile in filter(lambda x: x.endswith('.py'), FileCollection(root_name)):
        t = pyfile[root_name_length:-3].replace(path_sep, '.').replace('.__init__', '').replace('__init__', '')
        if t:
            yield t
