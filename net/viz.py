import re
from graphviz import Digraph


class DDigraph(Digraph):
    def edges(self, *args, **kwargs):
        super().edges(*args, **kwargs)
        return self


class _N:
    @staticmethod
    def return_none(*args, **kwargs):
        return None

    def __getattr__(self, item):
        return _N.return_none


_none = _N()


class Parse:
    class rx:
        lines = re.compile('\n|\r|\n\r|\r\n')
        nout_nin = re.compile(r'(\w+)\W+(\w+)')
        arrow = re.compile(r"([\w\s,]+)->([\w\s,]+)")
        wsc = re.compile(r"[\w\s,]+")
        csv = re.compile(r"[\s,]+")

    @staticmethod
    def loose_edges(s):
        return list(map(lambda x: x.groups(), filter(None, map(Parse.rx.nout_nin.search, Parse.rx.lines.split(s)))))

    @staticmethod
    def _arrow_edges_gen(s):
        for line in Parse.rx.lines.split(s):
            pipeline = Parse.rx.wsc.findall(line)
            pipeline = list(map(lambda x: Parse.rx.csv.split(x.strip()), pipeline))
            for nouts, nins in zip(pipeline[:-1], pipeline[1:]):
                for nout in nouts:
                    for nin in nins:
                        yield nout, nin

    @staticmethod
    def arrow_edges(s):
        return list(Parse._arrow_edges_gen(s))


def dagdisp(edges, edge_parser=Parse.arrow_edges, engine=None, **digraph_kwargs):
    """
    Make a Dag image flexibly.

    :param edges: The edges
    :param edge_parser: Function to parse a string into a list of edges
    :param engine:
        dot - "hierarchical" or layered drawings of directed graphs.
            This is the default tool to use if edges have directionality.
        neato - "spring model'' layouts.  This is the default tool to use if the graph is not too large (about 100 nodes)
            and you don't know anything else about it. Neato attempts to minimize a global energy function,
            which is equivalent to statistical multi-dimensional scaling.
        fdp - "spring model'' layouts similar to those of neato, but does this by reducing forces rather than
            working with energy.
        sfdp - multiscale version of fdp for the layout of large graphs.
        twopi - radial layouts, after Graham Wills 97. Nodes are placed on concentric circles depending their distance
            from a given root node.
        circo - circular layout, after Six and Tollis 99, Kauffman and Wiese 02.
            This is suitable for certain diagrams of multiple cyclic structures, such as certain telecommunications networks.
    :param digraph_kwargs: Other kwargs for graphviz.Digraph
    :return:
    """
    if isinstance(edges, str):
        edges = edge_parser(edges)
    return DDigraph(engine=engine, **digraph_kwargs).edges(edges)
