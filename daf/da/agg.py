__author__ = 'thor'


def with_do_to(df, with_col, do_fun, to_field):
    if isinstance(to_field, str):
        to_field = lambda x: to_field
    if isinstance(with_col, str):
        return {to_field(with_col): do_fun(df[with_col])}
    else:
        return {to_field(c): do_fun(df[c]) for c in with_col}
