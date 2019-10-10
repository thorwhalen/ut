"""
Functions and script to peek into the callable/argument structure of a set of callables, as given by
a module (or module dot-path string), class, callable, or list thereof.
"""
import matplotlib.pylab as plt
import importlib
import pandas as pd
import numpy as np
from typing import Callable, Iterator
import inspect
from types import ModuleType


class Required:
    def __repr__(self):
        return '_REQUIRED_'


_REQUIRED_ = Required()

callable_filt_for = {
    'callable': callable,
    'class': inspect.isclass,
    'function': inspect.isfunction,
    'function_or_class': lambda x: inspect.isclass(x) or inspect.isfunction(x)
}

DFLT_CALLABLE_FILT = 'callable'  # other choices: inspect.isfunction, inspect.isclass


def get_callable_filt(callable_filt):
    """
    Returns a boolean function that will be used to filter objects.
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :return:
    """
    if isinstance(callable_filt, str):
        callable_filt = callable_filt_for.get(callable_filt, None)
        if callable_filt is None:
            raise ValueError(f"No such callable_filt: {callable_filt}")
    return callable_filt


def module_classes(module):
    return filter(inspect.isclass, module.__dict__.values())


def callables_of_module(module, callable_filt=DFLT_CALLABLE_FILT):
    """

    :param module:
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :return:
    """
    callable_filt = get_callable_filt(callable_filt)
    if isinstance(module, str):
        module = importlib.import_module(module)
    return (x[1] for x in inspect.getmembers(module, predicate=callable_filt))


def callables_of_class(cls, callable_filt='function'):
    """

    :param cls:
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :return:
    """
    callable_filt = get_callable_filt(callable_filt)
    return (x[1] for x in inspect.getmembers(cls, predicate=callable_filt))


def get_callables_from(callables, callable_filt=DFLT_CALLABLE_FILT):
    """
    Get a generator of callable objects from the input x.
    :param callables: A module (or module dot-path string), class, callable, or list thereof
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :return: A generator of callable objects
    """
    callable_filt = get_callable_filt(callable_filt)
    if callable(callables):
        if not inspect.isclass(callables):
            yield from [callables]
        else:
            yield from get_callables_from(callables_of_class(callables), callable_filt=callable_filt)
    else:
        if isinstance(callables, str) or isinstance(callables, ModuleType):
            yield from get_callables_from(callables_of_module(callables), callable_filt=callable_filt)
        elif inspect.isclass(callables):
            yield callables
            yield from get_callables_from(callables_of_class(callables), callable_filt=callable_filt)
        else:
            try:
                for c in callables:
                    if not hasattr(c, '__qualname__'):
                        continue  # hack to avoid a case I don't understand
                    yield from get_callables_from(c, callable_filt=callable_filt)
            except:
                raise TypeError(f"Don't know how to handle the input: {c}\n"
                                "Wasn't a callable, string, module, or iterable of such.")


class SignatureExtractor:
    def __init__(self, normalize_var_names=True):
        def param_mint(param):
            if normalize_var_names:
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    d = {'name': '**kwargs'}
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    d = {'name': '*args'}
                else:
                    d = {'name': param.name}
            else:
                d = {'name': param.name}

            if param.default == inspect._empty:
                d['default'] = _REQUIRED_
            else:
                d['default'] = param.default

            return d

        self.param_mint = param_mint

    def __call__(self, obj):
        return [self.param_mint(p) for p in inspect.signature(obj).parameters.values()]


# Note: A function would make more sense, but chose to make a class to demo how classes are handled as callables
extract_name_and_default = SignatureExtractor(normalize_var_names=True)


def name_arg_default_dict_of_callables(callables: Iterator[Callable]) -> dict:
    """
    Get an {callable_name: {arg_name: arg_default, ...}, ...} dict from a collection of callables.
    See also: name_arg_default_dict_of_callables and arg_default_dict_of_module_classes
    :param callables: Iterable of callables
    :return: A dict
    """
    d = dict()
    for obj in callables:
        try:
            d[obj.__qualname__] = {x['name']: x['default'] for x in extract_name_and_default(obj)}
        except Exception as e:
            pass  # TODO: Give choice to warn instead of ignore
    return d


def arg_default_dict_of_callables(callables, callable_filt=DFLT_CALLABLE_FILT) -> dict:
    """
    Get an {callable_name: {arg_name: arg_default, ...}, ...} dict from a collection of callables taken from
    a specification of callables.
    :param callables = get_callables_from(callables, callable_filt=callable_filt)
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :return: A dict
    """
    callable_filt = get_callable_filt(callable_filt)
    return name_arg_default_dict_of_callables(get_callables_from(callables, callable_filt=callable_filt))


def non_null_counts(df: pd.DataFrame, null_val=np.nan):
    if null_val is np.nan:
        non_null_lidx = ~df.isna()
    else:
        non_null_lidx = df != null_val
    row_null_zero_count = non_null_lidx.sum(axis=1)
    col_null_zero_count = non_null_lidx.sum(axis=0)
    return row_null_zero_count, col_null_zero_count


def _df_of_callable_arg_default_dict(callable_arg_default_dict, null_fill='') -> pd.DataFrame:
    """
    Get a dataframe from a callable_arg_default_dict
    :param module:
    :param null_fill:
    :return:
    """
    d = pd.DataFrame.from_dict(callable_arg_default_dict)
    row_null_zero_count, col_null_zero_count = non_null_counts(d, null_val=np.nan)
    row_argsort = np.argsort(row_null_zero_count)[::-1]
    col_argsort = np.argsort(col_null_zero_count)[::-1]
    return d.iloc[row_argsort, col_argsort].fillna(null_fill).T


def callables_signatures_df(
        callables, callable_filt=DFLT_CALLABLE_FILT, null_fill='') -> pd.DataFrame:
    """
    Get a dataframe representing the signatures of the input callables.
    :param callables = get_callables_from(callables, callable_filt=callable_filt)
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :param null_fill: What to fill the empty cells with
    :return: A dataframe
    >>> import sys
    >>> df = callables_signatures_df(sys.modules[__name__])
    >>> list(df.columns.values)[:7]  # 7 most used argument names
    ['callable_filt', 'callables', 'self', 'module', 'null_fill', 'ylabel_left', 'X']
    >>> list(df.index.values)[:4]  # 4 functions with the most arguments
    ['heatmap', 'plot_nonnull_counts_of_signatures', 'heatmap_of_signatures', 'callables_signatures_df']
    """
    callable_filt = get_callable_filt(callable_filt)
    d = arg_default_dict_of_callables(callables, callable_filt=callable_filt)
    return _df_of_callable_arg_default_dict(d, null_fill=null_fill)


def heatmap(X, y=None, col_labels=None, figsize=None, cmap=None, return_gcf=False, ax=None,
            xlabel_top=True, ylabel_left=True, xlabel_bottom=True, ylabel_right=True, **kwargs):
    """
    Make a heatmap of a matrix or pandas.DataFrame, but let the function figure stuff out.
    """
    import pandas as pd
    import numpy as np
    n_items, n_cols = X.shape
    if col_labels is not None:
        if col_labels is not False:
            assert len(col_labels) == n_cols, \
                "col_labels length should be the same as the number of columns in the matrix"
    elif isinstance(X, pd.DataFrame):
        col_labels = list(X.columns)

    if figsize is None:
        x_size, y_size = X.shape
        if x_size >= y_size:
            figsize = (6, min(18, 6 * x_size / y_size))
        else:
            figsize = (min(18, 6 * y_size / x_size), 6)

    if cmap is None:
        if X.min(axis=0).min(axis=0) < 0:
            cmap = 'RdBu_r'
        else:
            cmap = 'hot_r'

    kwargs['cmap'] = cmap
    kwargs = dict(kwargs, interpolation='nearest', aspect='auto')

    if figsize is not False:
        plt.figure(figsize=figsize)

    if ax is None:
        plt.imshow(X, **kwargs)
    else:
        ax.imshow(X, **kwargs)
    plt.grid(None)

    if y is not None:
        y = np.array(y)
        assert all(sorted(y) == y), "This will only work if your row_labels are sorted"

        unik_ys, unik_ys_idx = np.unique(y, return_index=True)
        for u, i in zip(unik_ys, unik_ys_idx):
            plt.hlines(i - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
        plt.hlines(n_items - 0.5, 0 - 0.5, n_cols - 0.5, colors='b', linestyles='dotted', alpha=0.5)
        plt.yticks(unik_ys_idx + np.diff(np.hstack((unik_ys_idx, n_items))) / 2, unik_ys)
    elif isinstance(X, pd.DataFrame):
        y_tick_labels = list(X.index)
        plt.yticks(list(range(len(y_tick_labels))), y_tick_labels);

    if col_labels is not None:
        plt.xticks(list(range(len(col_labels))), col_labels)
    else:
        plt.xticks([])

    plt.gca().xaxis.set_tick_params(labeltop=xlabel_top, labelbottom=xlabel_bottom)
    plt.gca().yaxis.set_tick_params(labelleft=ylabel_left, labelright=ylabel_right)

    if return_gcf:
        return plt.gcf()


def heatmap_of_signatures(callables, callable_filt=DFLT_CALLABLE_FILT, figsize=None, cmap='gray_r'):
    """
    Visualize a matrix containing the all functions of the module, and their arguments.
    :param callables: A module (or module dot-path string), class, callable, or list thereof
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :param figsize: Size of the figure (will try to figure one out if not specified - pun noticed)
    :return:
    """
    callable_filt = get_callable_filt(callable_filt)
    callables = get_callables_from(callables, callable_filt=callable_filt)
    core_comp_df = callables_signatures_df(callables, callable_filt=callable_filt)
    t = (core_comp_df != '').astype(int) + (core_comp_df == _REQUIRED_).astype(int)
    heatmap(t, figsize=figsize, cmap=cmap)
    plt.xticks(rotation=90)
    plt.grid(False)
    plt.show()


def plot_nonnull_counts_of_signatures(
        callables, callable_filt=DFLT_CALLABLE_FILT, n_top_items=50, figsize=None, hspace=0.5):
    """
    For all callables of the input module, will plot (as a bar graph):
        The argument count of each callable
        The callable count of each argument (i.e., in how many callables are the given argument used)
    :param callables: A module (or module dot-path string), class, callable, or list thereof
    :param callable_filt: A boolean function or the strings 'callable', 'class', 'function', or 'function_or_class'
    :param n_top_items: Number of (top) items to plot
    :param figsize: Size of the figure (will try to figure one out if not specified - pun noticed)
    :param hspace: The space between both bar plots.
    :return:
    """
    callable_filt = get_callable_filt(callable_filt)
    callables = get_callables_from(callables, callable_filt=callable_filt)
    core_comp_df = callables_signatures_df(callables, callable_filt=callable_filt)
    t, tt = non_null_counts(core_comp_df, null_val='')
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    t.iloc[:n_top_items].plot(kind='bar')
    plt.subplot(2, 1, 2)
    tt.iloc[:n_top_items].plot(kind='bar')
    plt.subplots_adjust(hspace=hspace)
    plt.show()


if __name__ == '__main__':
    import argh

    argh.dispatch_commands([callables_signatures_df,
                            heatmap_of_signatures,
                            plot_nonnull_counts_of_signatures])
