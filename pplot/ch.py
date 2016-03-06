__author__ = 'thorwhalen'

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def force_axis_to_contain(axis=None, num=0):
    """
    changes axis limits so that it will contain num
    """
    fun = _axis_fun('lim', axis)
    lim = fun()
    if num < lim:
        fun([num, lim[1]])
    elif num > lim:
        fun([lim[0], num])


def force_origin():
    """
    changes axis limits so that it will contain the origin
    """
    force_axis_to_contain('x', 0)
    force_axis_to_contain('y', 0)


def axis_to_contain_data_and_padding(axis_obj=None, padding=0, axis=['x', 'y']):
    if isinstance(axis, list):
        for a in axis:
            axis_to_contain_data_and_padding(axis_obj, padding, a)
    else:
        d = _get_axis_axis(axis, axis_obj).get_data_interval()
        # t = _axis_fun('data_interval', axis, axis_obj=axis_obj)()
        _axis_fun('lim', axis)([d[0]*(1 - padding), d[1]*(1 + padding)])


def ratio_to_percent(axis=None, axis_obj=None):
    _get_axis_axis(axis=axis, axis_obj=axis_obj).set_major_formatter(FuncFormatter(_to_percent))


def _to_percent(y, position=None):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex']:
        return s + r'$\%$'
    else:
        return s + '%'


def _get_axis_axis(axis=None, axis_obj=None):
    axis_obj = axis_obj or plt.gca()
    if axis == 'y':
        return axis_obj.yaxis
    else:
        return axis_obj.xaxis


def _axis_fun(fun, axis=None, **kwargs):
    if fun == 'lim':
        if axis == 'y':
            fun = plt.ylim
        else:
            fun = plt.xlim
    # elif fun == 'data_interval':
    #     axis_obj = kwargs.get('axis_obj', plt.gca())
    #     if axis == 'y':
    #         fun = axis_obj.yaxis.get_data_interval
    #     else:
    #         fun = axis_obj.xaxis.get_data_interval
    else:
        NotImplementedError("Unknown fun")
    return fun


