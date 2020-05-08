import re
from string import Formatter

dflt_formatter = Formatter()


########### Partial and incremental formatting #########################################################################
class PartialFormatter(Formatter):
    """A string formatter that won't complain if the fields are only partially formatted.
    But note that you will lose the spec part of your template (e.g. in {foo:1.2f}, you'll loose the 1.2f
    if not foo is given -- but {foo} will remain).
    """

    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except KeyError:
            return '{' + key + '}'

    def format_fields_set(self, s):
        return {x[1] for x in self.parse(s) if x[1]}


partial_formatter = PartialFormatter()


# TODO: For those who love algorithmic optimization, there's some wasted to cut out here below.

def _unformatted(d):
    for k, v in d.items():
        if isinstance(v, str) and len(partial_formatter.format_fields_set(v)) > 0:
            yield k


def _fields_to_format(d):
    for k, v in d.items():
        if isinstance(v, str):
            yield from partial_formatter.format_fields_set(v)


def format_str_vals_of_dict(d, *, max_formatting_loops=10, **kwargs):
    """

    :param d:
    :param max_formatting_loops:
    :param kwargs:
    :return:

    >>> d = {
    ...     'filepath': '{root}/{file}.{ext}',
    ...     'ext': 'txt'
    ... }
    >>> format_str_vals_of_dict(d, root='ROOT', file='FILE')
    {'filepath': 'ROOT/FILE.txt', 'ext': 'txt'}

    Note that if the input mapping `d` and the kwargs have a conflict, the mapping version is used!

    >>> format_str_vals_of_dict(d, root='ROOT', file='FILE', ext='will_not_be_used')
    {'filepath': 'ROOT/FILE.txt', 'ext': 'txt'}

    But if you want to override an input mapping, you can -- the usual way:
    >>> format_str_vals_of_dict(dict(d, ext='will_be_used'), root='ROOT', file='FILE')
    {'filepath': 'ROOT/FILE.will_be_used', 'ext': 'will_be_used'}

    If you don't provide enough fields to satisfy all the format fields in the values of `d`,
    you'll be told to bugger off.

    >>> format_str_vals_of_dict(d, root='ROOT')
    Traceback (most recent call last):
    ...
    ValueError: I won't be able to complete that. You'll need to provide the values for:
      file

    And it's recursive...
    >>> d = {
    ...     'filepath': '{root}/{filename}',
    ...     'filename': '{file}.{ext}'
    ... }
    >>> my_configs = {'root': 'ROOT', 'file': 'FILE', 'ext': 'EXT'}
    >>> format_str_vals_of_dict(d, **my_configs)
    {'filepath': 'ROOT/FILE.EXT', 'filename': 'FILE.EXT'}

    # TODO: Could make the above work if filename is give, but not file nor ext! At least as an option.

    """
    d = dict(**d)  # make a shallow copy
    # The defaults (kwargs) cannot overlap with any keys of d, so:
    kwargs = {k: kwargs[k] for k in set(kwargs) - set(d)}

    provided_fields = set(d) | set(kwargs)
    missing_fields = set(_fields_to_format(d)) - provided_fields

    if missing_fields:
        raise ValueError("I won't be able to complete that. You'll need to provide the values for:\n" +
                         f"  {', '.join(missing_fields)}")

    for i in range(max_formatting_loops):
        unformatted = set(_unformatted(d))

        if unformatted:
            for k in unformatted:
                d[k] = partial_formatter.format(d[k], **kwargs, **d)
        else:
            break
    else:
        raise ValueError(f"There are still some unformatted fields, "
                         f"but I reached my max {max_formatting_loops} allowed loops. " +
                         f"Those fields are: {set(_fields_to_format(d)) - (set(d) | set(kwargs))}")

    return d


#######################################################################################################################

def compile_str_from_parsed(parsed):
    """The (quasi-)inverse of string.Formatter.parse.

    Args:
        parsed: iterator of (literal_text, field_name, format_spec, conversion) tuples,
        as yield by string.Formatter.parse

    Returns:
        A format string that would produce such a parsed input.

    >>> from string import Formatter
    >>> s =  "ROOT/{}/{0!r}/{1!i:format}/hello{:0.02f}TAIL"
    >>> assert compile_str_from_parsed(Formatter().parse(s)) == s
    >>>
    >>> # Or, if you want to see more details...
    >>> parsed = list(Formatter().parse(s))
    >>> for p in parsed:
    ...     print(p)
    ('ROOT/', '', '', None)
    ('/', '0', '', 'r')
    ('/', '1', 'format', 'i')
    ('/hello', '', '0.02f', None)
    ('TAIL', None, None, None)
    >>> compile_str_from_parsed(parsed)
    'ROOT/{}/{0!r}/{1!i:format}/hello{:0.02f}TAIL'
    """
    result = ''
    for literal_text, field_name, format_spec, conversion in parsed:
        # output the literal text
        if literal_text:
            result += literal_text

        # if there's a field, output it
        if field_name is not None:
            result += '{'
            if field_name != '':
                result += field_name
            if conversion:
                result += '!' + conversion
            if format_spec:
                result += ':' + format_spec
            result += '}'
    return result


def transform_format_str(format_str, parsed_tuple_trans_func):
    return compile_str_from_parsed(
        map(lambda args: parsed_tuple_trans_func(*args), dflt_formatter.parse(format_str)))


def _empty_field_name(literal_text, field_name, format_spec, conversion):
    if field_name is not None:
        return literal_text, '', format_spec, conversion
    else:
        return literal_text, field_name, format_spec, conversion


def auto_field_format_str(format_str):
    """Get an auto field version of the format_str

    Args:
        format_str: A format string

    Returns:
        A transformed format_str
    >>> auto_field_format_str('R/{0}/{one}/{}/{two}/T')
    'R/{}/{}/{}/{}/T'
    """
    return transform_format_str(format_str, _empty_field_name)


def _mk_naming_trans_func(names=None):
    if names is None:
        names = map(str, range(99999))
    _names = iter(names)

    def trans_func(literal_text, field_name, format_spec, conversion):
        if field_name is not None:
            return literal_text, next(_names), format_spec, conversion
        else:
            return literal_text, field_name, format_spec, conversion

    return trans_func


def name_fields_in_format_str(format_str, field_names=None):
    """Get a manual field version of the format_str

    Args:
        format_str: A format string
        names: An iterable that produces enough strings to fill all of format_str fields

    Returns:
        A transformed format_str
    >>> name_fields_in_format_str('R/{0}/{one}/{}/{two}/T')
    'R/{0}/{1}/{2}/{3}/T'
    >>> # Note here that we use the field name to inject a field format as well
    >>> name_fields_in_format_str('R/{foo}/{0}/{}/T', ['42', 'hi:03.0f', 'world'])
    'R/{42}/{hi:03.0f}/{world}/T'
    """
    return transform_format_str(format_str, _mk_naming_trans_func(field_names))


def match_format_string(format_str, s):
    """Match s against the given format string, return dict of matches.

    We assume all of the arguments in format string are named keyword arguments (i.e. no {} or
    {:0.2f}). We also assume that all chars are allowed in each keyword argument, so separators
    need to be present which aren't present in the keyword arguments (i.e. '{one}{two}' won't work
    reliably as a format string but '{one}-{two}' will if the hyphen isn't used in {one} or {two}).

    We raise if the format string does not match s.

    Author: https://stackoverflow.com/users/2593383/nonagon
    Found here: https://stackoverflow.com/questions/10663093/use-python-format-string-in-reverse-for-parsing

    Example:
    >>> fs = '{test}-{flight}-{go}'
    >>> s = fs.format(test='first', flight='second', go='third')
    >>> match_format_string(fs, s)
    {'test': 'first', 'flight': 'second', 'go': 'third'}
    """

    # First split on any keyword arguments, note that the names of keyword arguments will be in the
    # 1st, 3rd, ... positions in this list
    tokens = re.split(r'\{(.*?)\}', format_str)
    keywords = tokens[1::2]

    # Now replace keyword arguments with named groups matching them. We also escape between keyword
    # arguments so we support meta-characters there. Re-join tokens to form our regexp pattern
    tokens[1::2] = map(u'(?P<{}>.*)'.format, keywords)
    tokens[0::2] = map(re.escape, tokens[0::2])
    pattern = ''.join(tokens)

    # Use our pattern to match the given string, raise if it doesn't match
    matches = re.match(pattern, s)
    if not matches:
        raise Exception("Format string did not match")

    # Return a dict with all of our keywords and their values
    return {x: matches.group(x) for x in keywords}


def _is_not_none(x):
    return x is not None


def format_params_in_str_format(format_string):
    """
    Get the "parameter" indices/names of the format_string

    Args:
        format_string: A format string (i.e. a string with {...} to mark parameter placement and formatting

    Returns:
        A list of parameter indices used in the format string, in the order they appear, with repetition.
        Parameter indices could be integers, strings, or None (to denote "automatic field numbering".
    >>> format_string = '{0} (no 1) {2}, and {0} is a duplicate, {} is unnamed and {name} is string-named'
    >>> list(format_params_in_str_format(format_string))
    [0, 2, 0, None, 'name']
    """
    return map(lambda x: int(x) if str.isnumeric(x) else x if x != '' else None,
               filter(_is_not_none, (x[1] for x in dflt_formatter.parse(format_string))))


def n_format_params_in_str_format(format_string):
    """ The number of parameters"""
    return len(set(format_params_in_str_format(format_string)))


def arg_and_kwargs_indices(format_string):
    """

    Args:
        format_string: A format string (i.e. a string with {...} to mark parameter placement and formatting

    Returns:

    >>> format_string = '{0} (no 1) {2}, {see} this, {0} is a duplicate (appeared before) and {name} is string-named'
    >>> assert arg_and_kwargs_indices(format_string) == ({0, 2}, {'name', 'see'})
    >>> format_string = 'This is a format string with only automatic field specification: {}, {}, {} etc.'
    >>> arg_and_kwargs_indices(format_string)
    (None, None)
    """
    d = {True: set(), False: set()}
    for x in format_params_in_str_format(format_string):
        d[isinstance(x, int)].add(x)
    args_keys, kwargs_keys = _validate_str_format_arg_and_kwargs_keys(d[True], d[False])
    return args_keys, kwargs_keys


def _validate_str_format_arg_and_kwargs_keys(args_keys, kwargs_keys):
    """check that str_format is entirely manual or entirely automatic field specification"""
    if any(not x for x in kwargs_keys):  # {} (automatic field numbering) show up as '' in args_keys
        # so need to check that args_keys is empty and kwargs has only None (no "manual" names)
        if (len(args_keys) != 0) or (len(kwargs_keys) != 1):
            raise ValueError(
                f"cannot switch from manual field specification (i.e. {{number}} or {{name}}) "
                "to automatic (i.e. {}) field numbering. But you did:\n{str_format}")
        return None, None
    else:
        return args_keys, kwargs_keys


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


class PipelineTemplate(Formatter):
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
        ...         lambda x: list(map(lambda xx: xx + 2, x))
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
    return lambda x: list(map(func, x))


def templater(template):
    template = template.replace("{{}}", "{}")
    return template.format
