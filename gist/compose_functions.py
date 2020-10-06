from inspect import signature, Signature

# Note: A better version of this lives now on lined, an otosense repo and pip installable package
class Compose:
    def __init__(self, *funcs):
        """Performs function composition.
        That is, get a callable that is equivalent to a chain of callables.
        For example, if `f`, `h`, and `g` are three functions, the function
        ```
            c = Compose(f, h, g)
        ```
        is such that, for any valid inputs `args, kwargs` of `f`,
        ```
        c(*args, **kwargs) == g(h(f(*args, **kwargs)))
        ```
        (assuming the functions are deterministic of course).

        >>> def first(a, b=1):
        ...     return a * b
        >>>
        >>> def last(c) -> float:
        ...     return c + 10
        >>>
        >>> f = Compose(first, last)
        >>>
        >>> assert f(2) == 12
        >>> assert f(2, 10) == 30

        Let's check out the signature of f:

        >>> from inspect import signature
        >>>
        >>> assert str(signature(f)) == '(a, b=1) -> float'
        >>> assert signature(f).parameters == signature(first).parameters
        >>> assert signature(f).return_annotation == signature(last).return_annotation == float

        Border case: One function only

        >>> same_as_first = Compose(first)
        >>> assert same_as_first(42) == first(42)
        """
        self.funcs = funcs

        # Taking care of the signature...
        # Determining what the first and last function is.
        n_funcs = len(self.funcs)
        if n_funcs == 0:  # really, it would make sense that this is the identity, but we'll implement only when needed
            raise ValueError("You need to specify at least one function!")
        elif n_funcs == 1:
            first_func = last_func = funcs[0]
        else:
            first_func, *_, last_func = funcs
        # Finally, let's make the __call__ have a nice signature.
        # Argument information from first func and return annotation from last func
        self.__signature__ = Signature(signature(first_func).parameters.values(),
                                       return_annotation=signature(last_func).return_annotation)

    def __call__(self, *args, **kwargs):
        first_func, *other_funcs = self.funcs
        out = first_func(*args, **kwargs)
        for func in other_funcs:
            out = func(out)
        return out
