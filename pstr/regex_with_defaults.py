"""Get a `re.Pattern` instance (as given by re.compile()) with control over defaults of it's methods.
Useful to reduce if/else boilerplate when handling the output of search functions (match, search, etc.)
See [regex_search_hack.md](https://gist.github.com/thorwhalen/6c913e9be35873cea6efaf6b962fde07) for more explanatoins of the
use case.
Example;
>>> dflt_result = type('dflt_search_result', (), {'groupdict': lambda x: {}})()
>>> p = re_compile('.*(?P<president>obama|bush|clinton)', search=dflt_result, match=dflt_result)
>>>
>>> p.search('I am beating around the bush, am I?').groupdict().get('president', 'Not found')
'bush'
>>> p.match('I am beating around the bush, am I?').groupdict().get('president', 'Not found')
'bush'
>>>
>>> # if not match is found, will return 'Not found', as requested
>>> p.search('This does not contain a president').groupdict().get('president', 'Not found')
'Not found'
>>>
>>> # see that other non-wrapped re.Pattern methods still work
>>> p.findall('I am beating around the bush, am I?')
['bush']
"""

import re
from functools import wraps


def add_dflt(func, dflt_if_none):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is not None:
            return result
        else:
            if callable(dflt_if_none):
                return dflt_if_none()
            else:
                return dflt_if_none

    return wrapped_func


def re_compile(pattern, flags=0, **dflt_if_none):
    """Get a `re.Pattern` instance (as given by re.compile()) with control over defaults of it's methods.
    Useful to reduce if/else boilerplate when handling the output of search functions (match, search, etc.)
    Example;
    >>> dflt_result = type('dflt_search_result', (), {'groupdict': lambda x: {}})()
    >>> p = re_compile('.*(?P<president>obama|bush|clinton)', search=dflt_result, match=dflt_result)
    >>>
    >>> # trying p.search
    >>> p.search('I am beating around the bush, am I?').groupdict().get('president', 'Not found')
    'bush'
    >>> # trying p.match
    >>> p.match('I am beating around the bush, am I?').groupdict().get('president', 'Not found')
    'bush'
    >>>
    >>> # if not match is found, will return 'Not found', as requested
    >>> p.search('This does not contain a president').groupdict().get('president', 'Not found')
    'Not found'
    >>>
    >>> # see that other non-wrapped re.Pattern methods still work
    >>> p.findall('I am beating around the bush, am I?')
    ['bush']
    """
    compiled_regex = re.compile(pattern, flags=flags)
    intercepted_names = set(dflt_if_none)

    my_regex_compilation = type('MyRegexCompilation', (object,), {})()

    for _name, _dflt in dflt_if_none.items():
        setattr(my_regex_compilation, _name, add_dflt(getattr(compiled_regex, _name), _dflt))
    for _name in filter(lambda x: not x.startswith('__') and x not in intercepted_names,
                        dir(compiled_regex)):
        setattr(my_regex_compilation, _name, getattr(compiled_regex, _name))

    return my_regex_compilation
