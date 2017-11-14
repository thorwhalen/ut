from __future__ import division

import re
import string

pipe_split_p = re.compile("\s*\|\s*")
func_and_arg_p = re.compile('(?P<func>\w+)\((?P<args>.*)\)', flags=re.DOTALL)
comma_sep_p = re.compile('\s*,\s*')


def get_func_and_arg_dict(s):
    """
    Parses the input string recursively, returning:
        * if the string has the format f(*args): a nested func_and_arg_dict
        * else returns the string itself, unchanged.
    A func_and_arg_dict is a dict of the format:
        {"func": FUNC_STRING, "args": ARG_STRING_LIST},
    where ARG_STRING_LIST elements are themselves strings or func_and_arg_dicts.

    The intended use is to parse string.Formatter() spec strings and carry out the instructions therein.
    :param s: a string
    :return: the same string, or a {func:, args:} dict if the string has the f(*args) pattern
    >>> get_func_and_arg_dict("foo")
    'foo'
    >>> get_func_and_arg_dict("foo()")
    {'args': [], 'func': 'foo'}
    >>> get_func_and_arg_dict("foo(bar)")
    {'args': ['bar'], 'func': 'foo'}
    >>> get_func_and_arg_dict("f(g(x), y)")
    {'args': [{'args': ['x'], 'func': 'g'}, 'y'], 'func': 'f'}
    >>> get_func_and_arg_dict('f(g(x), "two words", h(z))')
    {'args': [{'args': ['x'], 'func': 'g'}, '"two words"', {'args': ['z'], 'func': 'h'}], 'func': 'f'}
    """
    match = func_and_arg_p.match(s)
    if match:
        func_and_arg_dict = match.groupdict()
        if 'args' in func_and_arg_dict:
            args_list = comma_sep_p.split(func_and_arg_dict['args'])
            if args_list == ['']:
                args_list = []
            for i, arg in enumerate(args_list):
                arg_expanded = get_func_and_arg_dict(arg)
                args_list[i] = arg_expanded
            func_and_arg_dict['args'] = args_list
        return func_and_arg_dict
    else:
        return s


class PipelineTemplate(string.Formatter):
    def __init__(self, **key_to_action):
        """
        A string.Formatter that accepts a |-separated specification of a pipeline through which the input value should
        go through before being output (cast automatically to a str).
        This formatter is created by specifying what functions correspond to the  names that will be used in the spec
        of the string. Standard format specifications (such as 04.04f, d, etc) can be used as well, anywhere in the
        pipeline.
        :param key_to_action: key=action specifications. Action must be a callable which will be applied to input value
        >>> p = PipelineTemplate(plus_ten=lambda x: x + 10,
        ...                      times_ten=lambda x: x * 10
        ... )
        >>> p.format('{x:plus_ten}', x=1)  # plus_ten points to the function that does that job
        '11'
        >>> p.format('{x:plus_ten|times_ten}', x=1)  # you can pipeline the functions you've specified
        '110'
        >>> p.format('{x} + 10 = {x:plus_ten}, ({x} + 10) * 10 = {x:plus_ten|times_ten}', x=1)
        '1 + 10 = 11, (1 + 10) * 10 = 110'
        >>> p.format('x + 10 = {0:plus_ten}, (x + 10) * 10 = {0:plus_ten|times_ten}', 1)  # no name use
        'x + 10 = 11, (x + 10) * 10 = 110'
        >>> p.format('{x: times_ten  | plus_ten  }', x=1)  # can have spaces between pipes
        '20'
        >>> p.format('{x:04.02f}', x=2)  # you can also use standard formatting specs
        '2.00'
        >>> p.format('{x:times_ten | plus_ten | 04.0f}', x=2)  # even in a pipeline
        '0030'
        >>> p = {
        ...     'f_wrap': lambda x: map('f({})'.format, x),
        ...     'csv': lambda x: ', '.join(x),
        ... }
        >>>
        >>> p = PipelineTemplate(**p)
        >>>
        >>> p.format('this --> {alist:f_wrap|csv}', alist=['A'])
        'this --> f(A)'
        >>> p.format('that --> {alist:f_wrap|csv}', alist=['A', 'B', 'C'])
        'that --> f(A), f(B), f(C)'
        >>> # and if you didn't define what you needed in the constructor arguments, you can always write python code
        >>> s = '''This {x:
        ...         lambda x: x + 2
        ...         | lambda x: x * 10
        ...         | 3.02f} was obtained through python functions.'''
        >>> PipelineTemplate().format(s, x=1)
        'This 30.00 was obtained through python functions.'
        >>> # and you can even use functions that need to be imported!
        >>> p = {
        ...     'add_10': lambda x: x + 10
        ... }
        >>> s = '''{x:
        ...         lambda x: map(lambda xx: xx + 2, x)
        ...         | lambda x: __import__('numpy').array(x) * 10
        ...         | __import__('numpy').sum
        ...         | add_10}'''
        >>> PipelineTemplate(**p).format(s, x=[1, 2])
        '80'
        """
        self.key_to_action = key_to_action

    def format_field(self, value, spec):
        spec = spec.strip()
        spec_list = pipe_split_p.split(spec)
        for spec in spec_list:
            try:
                if spec in self.key_to_action:
                    value = self.key_to_action[spec](value)
                else:
                    try:
                        f = eval(spec)
                        value = eval("f(value)")  # TODO: evals are not secure. Put safety checks in place.
                    except Exception:
                        value = super(PipelineTemplate, self).format_field(value, spec)
            except ValueError as e:
                raise ValueError("{}: {}".format(spec, e.args[0]))
        return str(value)


def wrapper(prefix='', suffix=''):
    return "{prefix}{{}}{suffix}".format(prefix=prefix, suffix=suffix).format


def mapper(func):
    return lambda x: map(func, x)


def templater(template):
    template = template.replace("{{}}", "{}")
    return template.format

