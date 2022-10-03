from functools import wraps
from typing import Any, Callable
from warnings import warn

raise NotImplementedError(
    '''
This module worked, almost, and then I broke it in an attempt to make it better. 
Don't use.
'''
)


def _first_arg(*args, **kwargs):
    if len(args) > 0:
        return args[0]
    else:
        try:
            return next(iter(kwargs.values()))
        except StopIteration:
            raise ValueError("There are no inputs: I can't get the first one!")


def if_first_arg_is_none_return_val(func, val=True):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if _first_arg(*args, **kwargs) is not None:
            return func(*args, **kwargs)
        else:
            return val

    return wrapped_func


def always_true(exc_type, exc_val, exc_tb):
    return True


def always_false(exc_type, exc_val, exc_tb):
    return False


class ExcCondition:
    """A medley of exception conditions (to be used with HandleExceptions instances)"""

    always_true = always_true
    always_false = always_false
    handle_all = always_true

    @staticmethod
    def from_exception_classes(*handled_exception_classes):
        def handle_exception(exc_type, exc_val, exc_tb):
            return issubclass(exc_type, handled_exception_classes)

        return if_first_arg_is_none_return_val(handle_exception, False)


class ExcCallback:
    """A medley of exception callbacks (to be used with HandleExceptions instances)"""

    always_true = always_true
    always_false = always_false
    ignore = always_true

    @staticmethod
    def raise_on_error(exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True
        else:
            return False

    @staticmethod
    def warn_and_ignore(msg=None, category=None, stacklevel=1, source=None):
        def exc_callback(exc_type, exc_val, exc_tb):
            nonlocal msg
            msg = msg or f'{exc_type}: {exc_val}'
            warn(msg, category=category, stacklevel=stacklevel, source=source)
            return True

        return if_first_arg_is_none_return_val(exc_callback, True)

    @staticmethod
    def print_and_raise(msg=None):
        def exc_callback(exc_type, exc_val, exc_tb):
            print(msg or f'{exc_type}: {exc_val}')
            return True

        return if_first_arg_is_none_return_val(exc_callback, True)


Traceback = Any
TypeValTbFunc = Callable[[type, Exception, Any], Any]


class HandleExceptions:
    """
    >>> t = 2
    >>> t
    2
    >>> print('hi')
    hi
    """

    conditions = ExcCondition
    callbacks = ExcCallback

    def __init__(
        self,
        condition: TypeValTbFunc,
        if_condition: TypeValTbFunc = ExcCallback.raise_on_error,
        if_not_condition: TypeValTbFunc = ExcCallback.ignore,
    ):
        self.condition = condition
        self.if_condition = if_condition
        self.if_not_condition = if_not_condition

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.condition(exc_type, exc_val, exc_tb):
            return self.if_condition(exc_type, exc_val, exc_tb)
        else:
            return self.if_not_condition(exc_type, exc_val, exc_tb)
        #
        #
        # if exc_type is None:
        #     return True
        # elif self.exc_condition(exc_type, exc_val, exc_tb):
        #     return self.exc_callback(exc_type, exc_val, exc_tb)
        # else:
        #     return False


class ModuleNotFoundWarning(HandleExceptions):
    """Will issue a warning when a ModuleNotFoundError is encountered.

    # TODO: doctest, when run, doesn't even find the tests!!!? What the!?! Figure out
    >>> with ModuleNotFoundWarning():
    ...     import collections
    >>> with ModuleNotFoundWarning():
    ...     import asdf
    /D/Dropbox/dev/p3/proj/ut/errors.py:143: UserWarning: It seems you don't have a required package.
      warn(self.msg)
    >>> with ModuleNotFoundWarning():
    ...     0 / 0
    Traceback (most recent call last):
      ...
    ZeroDivisionError: division by zero
    >>>
    """

    def __init__(self):
        super().__init__(
            condition=ExcCondition.from_exception_classes(ModuleNotFoundError),
            if_condition=ExcCallback.warn_and_ignore(),
        )


class IgnoreErrors(HandleExceptions):
    """Context manager that ignores specific error classes (and their sublcasses)

    >>> with IgnoreErrors(ZeroDivisionError):
    ...     print("all is fine here")
    all is fine here
    >>> with IgnoreErrors(ZeroDivisionError):
    ...     0 / 0  # should be ignored
    >>>
    >>> with IgnoreErrors(ZeroDivisionError):
    ...     assert False
    Traceback (most recent call last):
      ...
    AssertionError
    """

    def __init__(self, *error_classes):
        super().__init__(
            condition=ExcCondition.from_exception_classes(error_classes),
            if_condition=ExcCallback.ignore,
            if_not_condition=ExcCallback.raise_on_error,
        )
        self.error_classes = error_classes


class ExpectedError(RuntimeError):
    ...


class ExpectErrors(IgnoreErrors):
    """

    Allow ZeroDivisionError errors to happen, ignoring silently:
    >>> with ExpectErrors(ZeroDivisionError):
    ...     0/0

    Allow AssertionError and ValueError errors to happen, ignoring silently:
    >>> with ExpectErrors(AssertionError, ValueError):
    ...     raise ValueError("Some value error")
    ...     assert False


    >>> with ExpectErrors(AssertionError, ValueError):
    ...     raise ValueError("Some value error")
    ...     raise AssertionError("")
    ...     raise TypeError("")

    >>> with ExpectError(TypeError):
    ...     t = 3
    Traceback (most recent call last):
      ...
    NameError: name 'ExpectError' is not defined
    """

    """Context manager that expects some specific error classes (and their sublcasses),
    raising a ExpectedError if those errors don't happen. """

    def __exit__(self, exc_type, exc_val, exc_tb):
        expected_error_happened = super().__exit__(exc_type, exc_val, exc_tb)
        if not expected_error_happened:
            raise ExpectedError(
                'Expected one of these errors (or subclasses thereof) to be raised:'
                f'\n{self.error_classes}'
            )
        return expected_error_happened


class HandleExceptions:
    """
    >>> t = 2
    >>> t
    2
    >>> print('hi')
    hi
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.condition(exc_type, exc_val, exc_tb):
            return self.if_condition(exc_type, exc_val, exc_tb)
        else:
            return self.if_not_condition(exc_type, exc_val, exc_tb)


#
# class ModuleNotFoundErrorNiceMessage:
#     def __init__(self, msg=None):
#         self.msg = msg
#
#     def __enter__(self):
#         pass
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if exc_type is ModuleNotFoundError:
#             if self.msg is not None:
#                 warn(self.msg)
#             else:
#                 raise ModuleNotFoundError(f"""
# It seems you don't have required `{exc_val.name}` package for this Store.
# Try installing it by running:
#
#     pip install {exc_val.name}
#
# in your terminal.
# For more information: https://pypi.org/project/{exc_val.name}
#             """)
#
#
# class ModuleNotFoundWarning:
#     def __init__(self, msg="It seems you don't have a required package."):
#         self.msg = msg
#
#     def __enter__(self):
#         pass
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if exc_type is ModuleNotFoundError:
#             warn(self.msg)
#             return True
#
#
# class ModuleNotFoundIgnore:
#     def __enter__(self):
#         pass
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if exc_type is ModuleNotFoundError:
#             pass
#         return True
