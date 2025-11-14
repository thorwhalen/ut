r"""
A module to get extractors from regular expressions more easily.

For example:

>>> import re, collections
>>> class Extractors(Rx):
...     obfuscate_clients = (
...           re.compile("Google|Apple|LinkedIn", re.IGNORECASE),
...           lambda p: lambda x: p.sub('***', x))
...     top_tokens = '\w+', '.findall', X, collections.Counter, lambda x: x.most_common(5)
>>>
>>> rx = Extractors()
>>> rx.obfuscate_clients('We do not want to mention clients like Google, apple, or linkedin by name')
'We do not want to mention clients like ***, ***, or *** by name'
>>> rx.top_tokens("I do not like green eggs and ham. I do not like them Sam-I-am!")
[('I', 3), ('do', 2), ('not', 2), ('like', 2), ('green', 1)]
"""
import re
from collections.abc import Callable
from re import Pattern
from operator import itemgetter, attrgetter
from functools import reduce


def compose(*functions):
    def compose2(f, g):
        return lambda x: f(g(x))

    return reduce(compose2, functions, lambda x: x)


def _is_pipeline_spec(obj):
    return (
        isinstance(obj, tuple)
        and len(obj) > 0
        and callable(obj[0])
        and getattr(obj[0], '__qualname__', '').startswith('Pattern.')
    )  # TODO: less hacky way?


class FuncPipe:
    def __init__(self, *funcs):
        assert FuncPipe.is_valid_pipeline_spec(
            funcs
        ), 'All inputs of FuncPipe must be callable'
        self.funcs = funcs

    def __call__(self, *args, **kwargs):
        output = self.funcs[0](*args, **kwargs)
        for f in self.funcs[1:]:
            output = f(output)
        return output

    @classmethod
    def is_valid_pipeline_spec(cls, obj):
        return isinstance(obj, tuple) and len(obj) > 0 and all(callable(o) for o in obj)


# TODO: find less hacky way?
def is_pattern_method(obj):
    return callable(obj) and getattr(obj, '__qualname__', '').startswith('Pattern.')


class Spec:
    def __init__(self, val):
        self.val = val


class Input:
    """An object to indicate that the value """

    pass


X = Input()


class Literal:
    """An object to indicate that the value should be considered literally"""

    def __init__(self, val):
        self.val = val


# TODO: Consider a function builder instead
class RxPipe:
    short_hands = {
        '_groups': lambda x: x.groups() if x is not None else None,
        '_group0': lambda x: x.group(0) if x is not None else None,
        '_group1': lambda x: x.group(1) if x is not None else None,
        '_remove': lambda pattern: lambda x: pattern.sub('', x),
    }

    def __init__(self, pattern, *steps):
        """
        The pattern can be a str, a re.Pattern object (what you get with re.compile(str)),
        or a "pattern method" (what you get with re.compile(str).method)
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        self.pattern = pattern
        self.steps = steps

        if is_pattern_method(pattern):
            pass
        assert isinstance(pattern, (str, Pattern))
        # assert FuncPipe.is_valid_pipeline_spec(post_funcs), "All inputs of FuncPipe must be callable"
        self._input_steps = steps
        self.steps = tuple(map(self.mk_func, self._input_steps))
        self.pipeline = compose(*self.steps)
        self._methods = set(dir(self))

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        if attr in self._methods:
            return getattr(self, attr)
        else:
            return getattr(self.pattern, attr)

    @classmethod
    def is_pipeline_tuple(cls, x):
        return isinstance(x, tuple) and len(x) >= 1 and isinstance(x[0], (str, Pattern))

    # PATTERN: tree crud pattern
    def mk_func(self, func):
        if isinstance(func, str):  # if it's a string, try three things
            if func in self.short_hands:  # look in short hands first
                func = self.short_hands[func]
            elif func.startswith('.'):
                func = attrgetter(func[1:])  # TODO: enable dot paths
            elif func.startswith('['):
                assert func.endswith(']'), 'If you start with [, you need to end with ]'
                key = func[1:-1]
                if key:
                    func = itemgetter(key)
                else:
                    func = attrgetter('__getitem__')
            else:
                raise ValueError(
                    f"The string func specification couldn't be resolved: {func}. "
                    f'If an actual string value is what you were going for, use `Literal(your_string)`'
                )
        # assert callable(func), f"func wasn't callable: {func}"
        return func

    def __repr__(self):
        if len(self.steps) == 0:
            return f'{self.__class__.__name__}({self.pattern})'
        else:
            quote = lambda x: f"'{x}'" if isinstance(x, str) else x
            return (
                f'{self.__class__.__name__}'
                + f'({self.pattern}'
                + f"{', ' + ', '.join(quote(x) for x in self._input_steps)})"
            )

    def __call__(self, input_val):
        cumul = self.pattern
        input_val_already_used = False
        for s in self.steps:
            if callable(s):
                cumul = s(cumul)
            else:
                if s != X:
                    cumul = cumul(s)
                else:
                    cumul = cumul(input_val)
                    input_val_already_used = True
        if not input_val_already_used:
            return cumul(input_val)
        else:
            return cumul


class Rx:
    r"""
    A class to make regular-expression based extractors.

    >>> import re
    >>>
    >>>
    >>> from ut.parse.rx import Rx, RxPipe, X
    >>>
    >>> class Extractors(Rx):
    ...     alphanum = '\w+'  # strings will be wrapped with re.compile
    ...     numbers = ('[\.\d]+', '.findall')  # tuples are pipelines
    ...     clients = re.compile("Google|Apple|LinkedIn", re.IGNORECASE), '_remove'  # don't need parens
    ...     simple_email = '[\w\.]+@\w+\.\w+', '.search', X, '.group', 0
    ...
    >>> rx = Extractors()

    A string `s` is converted to a re.compile(s) object.
    >>> rx.alphanum  # a string s is converted to a re.compile(s) object
    RxPipe(re.compile('\\w+'))
    >>> assert rx.alphanum.findall('here are tokens') == ['here', 'are', 'tokens']

    Tuples that start with a string or a re.compile object are intepreted as a pipeline.
    A pipeline string element starting with a '.' corresponds to getting an attribute.

    >>> rx.numbers('1, 11, and 19 are numbers, and so is 3.1415')
    ['1', '11', '19', '3.1415']

    A pipeline string element that is listed in the interpreter's short_hand dict will be replaced by it's value.
    Here, that value is `lambda pattern: lambda x: pattern.sub('', x)`, so...

    >>> rx.clients('We do not want to mention clients like Google, apple, or linkedin by name')
    'We do not want to mention clients like , , or  by name'

    The `X` of the module is used to mark where the input value of the pipeline should be inserted
    (in case it's not at the very end of the tuple (i.e. the beginning of the pipeline)).
    Also, any objects of the pipeline tuple that weren't interpreted will be left as is.

    So the pipeline:
    ```
        '[\w\.]+@\w+\.\w+', '.search', X, '.group', 0
    ```
    results a function equivalent to
    ```
        lambda x: re.compile('[\w\.]+@\w+\.\w+').search(x).group(0)
    ```
    >>> rx.simple_email("No name@email.com here!")
    'name@email.com'

    Want to extend this? For example, want to use your own "short_hands" mappings from string to pipeline element?
    Here's an example how.

    Say you want to add.
    >>> from collections import Counter
    >>> import re
    >>>
    >>> class MyInterpreter(RxPipe):
    ...         short_hands = dict(RxPipe.short_hands,  # keep the short_hands of RxPipe (or not), but add one:
    ...                      replace_by_three_stars = lambda pattern: lambda x: pattern.sub('***', x)
    ...                      )
    ...
    >>> class Extractors(Rx, interpreter=MyInterpreter):
    ...     obfuscate_clients = re.compile("Google|Apple|LinkedIn", re.IGNORECASE), 'replace_by_three_stars'
    ...     top_tokens = '\w+', '.findall', X, Counter, lambda x: x.most_common(5)
    >>>
    >>> rx = Extractors()
    >>> rx.obfuscate_clients('We do not want to mention clients like Google, apple, or linkedin by name')
    'We do not want to mention clients like ***, ***, or *** by name'
    >>> rx.top_tokens("I do not like green eggs and ham. I do not like them Sam-I-am!")
    [('I', 3), ('do', 2), ('not', 2), ('like', 2), ('green', 1)]

    """

    def __init_subclass__(cls, interpreter=RxPipe, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr_name in (a for a in dir(cls) if not a.startswith('__')):
            attr_obj = getattr(cls, attr_name)
            if interpreter.is_pipeline_tuple(attr_obj):
                setattr(cls, attr_name, interpreter(*attr_obj))
            elif isinstance(attr_obj, str):
                setattr(cls, attr_name, interpreter(attr_obj))
            elif isinstance(attr_obj, Literal):
                setattr(cls, attr_name, attr_obj.val)

    Literal = Literal  # just to have Literal available as Rx.Literal
