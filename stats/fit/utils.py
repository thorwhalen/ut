"""Utils for training models"""
__author__ = 'thor'

from numpy import *
import re
import inspect


def nrmse(ydata, ypred):
    return sqrt(mean((ypred - ydata) ** 2)) / (max(ydata) - min(ydata))


def cv_rmse(ydata, ypred):
    return sqrt(mean((ypred - ydata) ** 2)) / mean(ydata)


def formula_str(fit_func, params, num_format_or_precision=2):
    """
    Returns a string showing what a fit functions' formula is, with params injected.
    The normal use is when you have a fit function, say:
        def func(x, a, b):
            return a + b * x)
    which you then fit to data
        params, pcov = curve_fit(func, xdata, ydata, p0=(1.0, 1.0))
    to get some numerical params, say:
        params == [1.07815647e-06,  1.28497311e+00]

    Then if you call
        formula_str(func, params)
    you'll get
        1.07e-06 + 1.28e+00 * x

    You can control the appearance of the numerical params in the formula using the num_format_or_precision argument.

    Note that to work, the formula of the fit function has to fit completely in the return line of the function.
    """
    if not isinstance(num_format_or_precision, str):
        num_format_or_precision = "{:." + str(num_format_or_precision) + "e}"

    # get the param names from the code of the function
    param_names = re.compile('def [^(]+\(([^)]+)\)').findall(inspect.getsource(fit_func))[0].split(', ')[1:]

    # get the formula string from the code of the function
    formula_str = re.compile('return ([^\n]*)').findall(inspect.getsource(fit_func))[0]

    # replace the param values in the formula string to get the result
    rep = dict((re.escape(k), num_format_or_precision.format(v)) for k, v in zip(param_names, params))
    pattern = re.compile("|".join(list(rep.keys())))
    formula_str_with_nums = pattern.sub(lambda m: rep[re.escape(m.group(0))], formula_str)

    return formula_str_with_nums