import networkx as nx
from graphviz import Graph, Digraph
from collections import Counter
import pandas as pd

from matplotlib.pylab import figure, tight_layout, subplots

__author__ = 'thor'


def mk_partite_graph_from_df(df, cols=None, draw=True, figsize=(10, 20), **kwargs):
    kwargs = dict({"alpha": 0.2, "with_labels": True}, **kwargs)
    if cols is None:
        cols = df.columns
    elif isinstance(cols, int):
        cols = df.columns[:cols]

    B = nx.Graph()
    df = df[cols].dropna().drop_duplicates()

    partite_set = list()
    for i, col in enumerate(cols):
        this_partite_set = df.groupby(col).size().sort_values(ascending=True).index.values
        partite_set.append(this_partite_set)
        B.add_nodes_from(this_partite_set, bipartite=i)

    for i in range(len(cols) - 1):
        B.add_edges_from(list(df.iloc[:, [i, i + 1]].to_records(index=False)))

    if draw:
        pos = {}
        for i, this_partite_set in enumerate(partite_set):
            pos.update((node, (i, index / len(this_partite_set))) for index, node in enumerate(this_partite_set))

        figure(figsize=figsize)
        nx.draw(B, pos=pos, **kwargs)


def dict_to_attr_str(d: dict):
    """Get an graphviz attribute string from a dict
    >>> assert dict_to_attr_str(d) == '[label="tik tok" score="42"]'
    """
    if d is not None and len(d) > 0:
        return '[' + ' '.join(f'{k}="{v}"' for k, v in d.items()) + ']'
    else:
        return ''


def mk_most_common_indexing(arr):
    return {x: i for i, (x, _) in enumerate(Counter(arr).most_common())}


def mk_indexings(df, col_indexer=mk_most_common_indexing, prefix_with_col=True):
    index_for = {col: col_indexer(df[col]) for col in df.columns}  # numerical indexing
    if prefix_with_col:
        # ... to which we add a prefix corresponding to the column name so we can distinguish between columns
        index_for = {k: {kk: f"{k}_{vv}" for kk, vv in v.items()} for k, v in index_for.items()}
    return index_for


def apply_col_specific_func(df, func_for_col):
    return pd.concat((df[col].apply(func_for_col[col]) for col in df.columns), axis=1)


def apply_indexing(df, indexing_for_column: dict):
    func_for_col = {col: indexing_for_column[col].get for col in df.columns}
    return apply_col_specific_func(df, func_for_col)


def graphviz_body_gen(df, from_node, to_node, edge_label=None,
                      directed=False, from_node_attrs=None, to_node_attrs=None, edge_attrs=None):
    index_for = mk_indexings(df)
    dfi = apply_indexing(df, index_for)

    edge_chr = '->' if directed else '--'
    from_node_attrs = from_node_attrs or {}
    to_node_attrs = to_node_attrs or {}
    edge_attrs = edge_attrs or {}

    for lbl, idx in zip(df.to_dict(orient='records'), dfi.to_dict(orient='records')):
        attr_str = dict_to_attr_str(
            dict(edge_attrs, label=f"{lbl[edge_label]}" if (edge_label is not None) else ''))
        yield f"{idx[from_node]} {edge_chr} {idx[to_node]} {attr_str}"

        attr_str = dict_to_attr_str(dict(from_node_attrs, label=f"{lbl[from_node]}"))
        yield f"{idx[from_node]} {attr_str}"

        attr_str = dict_to_attr_str(dict(to_node_attrs, label=f"{lbl[to_node]}"))
        yield f"{idx[to_node]} {attr_str}"


def df_to_graphviz(df, from_node, to_node, edge_label=None,
                   directed=False, from_node_attrs=None, to_node_attrs=None, edge_attrs=None,
                   **graph_kwargs):
    body = list(graphviz_body_gen(df, from_node, to_node, edge_label=edge_label,
                                  directed=directed, from_node_attrs=from_node_attrs,
                                  to_node_attrs=to_node_attrs, edge_attrs=edge_attrs))
    mk_graph = Digraph if directed else Graph
    graph_kwargs.update(body=body)
    return mk_graph(**graph_kwargs)
