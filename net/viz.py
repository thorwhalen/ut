from graphviz import Digraph


class DDigraph(Digraph):
    def edges(self, *args, **kwargs):
        super().edges(*args, **kwargs)
        return self


class Parse:
    import re

    lines_p = re.compile('\n|\r|\n\r|\r\n')
    nout_nin_p = re.compile(r'(\w+)\W+(\w+)')

    @staticmethod
    def loose_edges(s):
        return list(map(lambda x: x.groups(), filter(None, map(Parse.nout_nin_p.search, Parse.lines_p.split(s)))))

    @staticmethod
    def arrow_edges(s):
        list(map(lambda x: x.split(' -> '), filter(lambda x: x, s.split('\n'))))


def dagdisp(s, edge_parser=Parse.loose_edges):
    return DDigraph().edges(edge_parser(s))
