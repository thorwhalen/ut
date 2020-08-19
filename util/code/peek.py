from inspect import getsource, getsourcelines, signature
from typing import Any, Optional


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
