from __future__ import division

import networkx as nx
from matplotlib.pylab import figure

__author__ = 'thor'


def mk_bipartite_graph_from_df(df, cols=None, draw=True):
    if cols is None:
        cols = df.columns[:2]

    B = nx.Graph()
    df = df[cols].dropna()
    b0 = df.groupby(cols[0]).size().sort_values(ascending=True).index.values
    b1 = df.groupby(cols[1]).size().sort_values(ascending=True).index.values
    B.add_nodes_from(b0, bipartite=0)  # Add the node attribute "bipartite"
    B.add_nodes_from(b1, bipartite=1)
    B.add_edges_from(list(df.to_records(index=False)))

    if draw:
        pos = {}
        pos.update((node, (1, index / len(b0))) for index, node in enumerate(b0))
        pos.update((node, (2, index / len(b1))) for index, node in enumerate(b1))
        figure(figsize=(10, 20))
        nx.draw(B, pos=pos, with_labels=True, alpha=0.2)