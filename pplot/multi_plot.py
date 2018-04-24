from __future__ import division

import matplotlib.pylab as plt


def _get_attr_args_and_kwargs_from_ax_call_item(ax_call_item):
    if isinstance(ax_call_item, dict):
        attr = ax_call_item['attr']
        args = ax_call_item.pop('args', ())
        kwargs = ax_call_item.pop('kwargs', {})
    else:
        attr = ax_call_item[0]
        if len(ax_call_item) == 2:
            if isinstance(ax_call_item[1], dict):
                args = ()
                kwargs = ax_call_item[1]
            else:
                args = ax_call_item[1]
                kwargs = {}
        else:
            args, kwargs = ax_call_item[1:]
    return attr, args, kwargs

def multi_row_plot(plot_func=plt.plot, data_list=(), figsize=3, plot_func_kwargs=None, ax_calls=()):
    """
    Quickly plotting multiple rows of data.

    :param plot_func: The plotting function to use.
    :param data_list: The list of datas to plot. For each "row_data" of data_list, a row will be created and plot_func
        will be called, using that item as input. If row_data is:
            * a dict, plot_func(**dict(plot_func_kwargs, **row_data)) will be called to populate that row
            * a tuple, plot_func(*row_data, **plot_func_kwargs) will be called to populate that row
            * if not, plot_func(row_data, **plot_func_kwargs) will be called to populate that row
    :param figsize: The figsize to use. If
        * a tuple of length 2, figure(figsize=figsize) will be called to create the figure
        * a number (int or float), figure(figsize=(16, n_rows * figsize_units_per_row)) will be called
        * If None, figure won't be called (we assume therefore, it's been created already, for example
    :param plot_func_kwargs: The kwargs to use as arguments of plot_func for every data row.
    :param ax_calls: A list of (attr, args, kwargs) triples that will result in calling
            getattr(ax, attr)(*args, **kwargs)
        for every ax in ax_list (the list of row axes)
    :return: ax_list, the list of axes for each row
    """
    if plot_func_kwargs is None:
        plot_func_kwargs = {}
    n_rows = len(data_list)
    if isinstance(figsize, (int, float)):
        figsize_units_per_row = figsize
        figsize = (16, n_rows * figsize_units_per_row)
    if isinstance(figsize, (tuple, list)) and len(figsize) == 2:
        plt.figure(figsize=figsize)

    ax_list = list()
    for row_idx, row_data in enumerate(data_list, 1):
        #         print(row_data)
        plt.subplot(n_rows, 1, row_idx)
        specific_ax_calls = ()
        if isinstance(row_data, dict):
            specific_ax_calls = row_data.pop('ax_calls', ())
            if 'row_data' in row_data:
                row_data = row_data['row_data']

        if isinstance(row_data, dict):
            plot_func(**dict(plot_func_kwargs, **row_data))
        elif isinstance(row_data, tuple):
            plot_func(*row_data, **plot_func_kwargs)
        else:
            plot_func(row_data, **plot_func_kwargs)

        ax = plt.gca()
        for attr, args, kwargs in map(_get_attr_args_and_kwargs_from_ax_call_item, specific_ax_calls):
            getattr(ax, attr)(*args, **kwargs)

        ax_list.append(ax)

    for ax in ax_list:
        for attr, args, kwargs in map(_get_attr_args_and_kwargs_from_ax_call_item, ax_calls):
            getattr(ax, attr)(*args, **kwargs)

    return ax_list
