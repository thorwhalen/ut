

__author__ = 'thor'

import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy import array

from matplotlib.colors import rgb2hex
from matplotlib import cm

from bokeh.plotting import figure, output_file
from bokeh.models import HoverTool, ColumnDataSource


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cityblock


def _distance(df, dist_fun, sort=None):
    row_name = df.index.name or 'row'
    col_name = df.columns.name or 'col'
    cumul = list()
    for row, col in itertools.product(df.index.values, df.index.values):
        cumul.append({row_name: row, col_name: col, 'val': dist_fun(df.loc[row], df.loc[col])})
    cumul = pd.DataFrame(cumul)
    cumul = cumul.set_index([col_name, row_name]).unstack(col_name)['val']
    if sort is not None:
        if sort == 'val_sum':
            var_order = [x for (y,x) in sorted(zip(cumul.sum().as_matrix(), cumul.index.values))]
            cumul = cumul[var_order].loc[var_order]
    return cumul


def _mk_distance_matrix(dd,
                        dist_fun=cityblock,
                        n_clus=1,
                        linkage_method='complete',
                        clus_criterion='maxclust'):
    ddd = dd / dd.sum()  # normalize the observations
    ddd = _distance(ddd, dist_fun=dist_fun)  # make the distance matrix (between each observation)

    Z = linkage(dd, method=linkage_method, metric=dist_fun)

    idx_df = pd.DataFrame({'clus_idx': array(fcluster(Z, t=n_clus, criterion=clus_criterion)),
                           'distance_sum': ddd.sum().values})
    idx = idx_df.sort_values(by=['clus_idx', 'distance_sum'], axis=0, ascending=True).index.values

    return ddd.iloc[idx, idx]


def compute_and_display_distances(dd,
                                  distance_matrix_fun=_mk_distance_matrix,
                                  cmap=cm.gray_r,
                                  data_to_01_color=None,
                                  graph_title="",
                                  output_filepath='bokeh_distance_matrix.html'):

    dist_mat = distance_matrix_fun(dd)

    # input handling
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    if data_to_01_color is None:
        min_val = np.min(dist_mat.as_matrix())
        max_val = np.max(dist_mat.as_matrix())
        val_range = max_val - min_val

        def data_to_01_color(x):
            return (x - min_val) / val_range

    data = _square_df_to_bokeh_graph(dist_mat)

    names, source = _mk_distmat_name_and_source(cmap, data, data_to_01_color)

    output_file(output_filepath)


    p = _mk_matrix_bokeh_fig(graph_title, names, source)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        ('y,x=', '@yname, @xname'),
        ('', '@ydata, @xdata'),
        ('val', '@vals'),
        ('what', '@poo')
    ])

    return p


def _mk_matrix_bokeh_fig(graph_title, names, source):
    p = figure(title=graph_title,
               x_axis_location="above", tools="resize,hover,save",
               x_range=list(reversed(names)), y_range=names)
    p.plot_width = 800
    p.plot_height = 800
    p.rect('xname', 'yname', 0.9, 0.9, source=source,
           color='colors', alpha='alphas', line_color=None)
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "5pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3
    return p


def _mk_distmat_name_and_source(cmap, data, data_to_01_color):
    nodes = data['nodes']
    names = [node['name'] for node in nodes]
    # names = [node['name'] for node in sorted(data['nodes'], key=lambda x: x['group'])]
    N = len(nodes)
    vals = np.zeros((N, N))
    for link in data['links']:
        vals[link['source'], link['target']] = link['value']
        vals[link['target'], link['source']] = link['value']
    xname = []
    yname = []
    color = []
    alpha = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            xname.append(n1['name'])
            yname.append(n2['name'])
            color_spec = cmap(data_to_01_color(vals[i, j]))
            alpha.append(color_spec[3])
            color.append(rgb2hex(color_spec[:3]))
    source = ColumnDataSource(
        data=dict(
            xname=xname,
            yname=yname,
            colors=color,
            alphas=alpha,
            vals=vals.flatten()
        )
    )
    return names, source


def _mk_distmat_name_and_source_including_data(cmap, data, data_to_01_color):
    nodes = data['nodes']
    names = [node['name'] for node in nodes]
    # names = [node['name'] for node in sorted(data['nodes'], key=lambda x: x['group'])]
    N = len(nodes)
    vals = np.zeros((N, N))
    for link in data['links']:
        vals[link['source'], link['target']] = link['value']
        vals[link['target'], link['source']] = link['value']
    xname = []
    yname = []
    xdata = []
    ydata = []
    color = []
    alpha = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            xname.append(n1['name'])
            yname.append(n2['name'])
            xdata.append(data['df'].loc[xname].to_string())
            ydata.append(data['df'].loc[yname].to_string())
            color_spec = cmap(data_to_01_color(vals[i, j]))
            alpha.append(color_spec[3])
            color.append(rgb2hex(color_spec[:3]))
    source = ColumnDataSource(
        data=dict(
            xname=xname,
            yname=yname,
            colors=color,
            alphas=alpha,
            vals=vals.flatten(),
            xdata=xdata,
            ydata=ydata
        )
    )
    return names, source


def square_df_heatmap(df,
                      cmap=cm.gray_r,
                      data_to_01_color=None,
                      graph_title="",
                      output_filepath='bokeh_heatmap.html'):

    # input handling
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)

    if data_to_01_color is None:
        min_val = np.min(df.as_matrix())
        max_val = np.max(df.as_matrix())
        val_range = max_val - min_val

        def data_to_01_color(x):
            return (x - min_val) / val_range

    data = _square_df_to_bokeh_graph(df)

    names, source = _mk_distmat_name_and_source(cmap, data, data_to_01_color)

    output_file(output_filepath)

    p = _mk_matrix_bokeh_fig(graph_title, names, source)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict([
        ('y,x=', '@yname, @xname'),
        ('val', '@vals'),
    ])

    return p


def _square_df_to_bokeh_graph(df):
    data = dict()
    node_names = df.index.values
    n = len(node_names)
    data['nodes'] = [{'name': v} for i, v in enumerate(node_names)]
    data['links'] = [{'source': source, 'target': target, 'value': df.iloc[source, target]}
                     for source, target in itertools.product(list(range(n)), list(range(n)))]
    # data['df'] = df
    return data