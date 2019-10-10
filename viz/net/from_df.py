

import networkx as nx
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