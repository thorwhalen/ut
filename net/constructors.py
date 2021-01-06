from itertools import chain, starmap, product, repeat
from typing import Mapping, Generator, Iterable


def pair_count_sr_to_coo_matrix(sr):
    from scipy import sparse

    """ Takes a (i,j)->count series and makes a sparse (weighted) adjacency matrix (coo_matrix) out of it """
    data, i, j = list(zip(*[(x[1], x[0][0], x[0][1]) for x in iter(sr.items())]))
    return sparse.coo_matrix((data, (i, j)))


def fanout_v(k, v):
    return zip(repeat(k), v)  # equivalently, product([k], v), but zip(repeat slightly faster)


def mapkv(kvfunc, mapping):
    return starmap(kvfunc, mapping.items())


def kv_fanout(mapping):
    return chain.from_iterable(mapkv(fanout_v, mapping))


def adjacencies_to_edges(adjacencies: Mapping) -> Generator:
    """
    A generator of edges taken from the input adjacencies.

    https://en.wikipedia.org/wiki/Adjacency_list
    :param adjacencies: A Mapping. adjacencies[from_node] is an iterable of to_nodes (possibly empty).
        Whether the edge is directed, or which direction it has (from_node->to_node, or visa versa)
        is irrelevant to the functioning of the generator.
    :return: A generator of (from_node, to_node) edges
    """
    return kv_fanout(adjacencies)  # just an alias of kv_fanout


from collections import defaultdict
from functools import cached_property  # need python 3.8 (or find backport online)

# Class to represent a graph
class Graph:
    def __init__(self, adjacencies=None):
        self.g = adjacencies or defaultdict(list)  # dictionary containing adjacency List

    def _iter_edges(self):
        return product([])
    def __iter__(self):
        return self.edges

    # @cached_property
    # def edges(self):

    # function to add an edge to graph
    def add_edge(self, u, v):
        self.g[u].append(v)

    def display(self):
        import graphviz
        return graphviz.Digraph(
            body=[f"{source} -> {target}" for source, target in adjacencies_to_edges(self.g)])
    # A recursive function used by topologicalSort


    def _helper(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.g[v]:
            if not visited[i]:
                self._helper(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    def topological_sort(self):
        # Mark all the vertices as not visited
        visited = [False] * len(self.g)
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        print(self.g)
        for i in range(len(self.g)):
            if not visited[i]:
                self._helper(i, visited, stack)
                # print(f"{i=}, {visited=}, {stack=}")
        return stack

    def _edges_from_vertex(self, v, already_visited):

        # Mark the current node as visited.
        already_visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.g[v]:
            if not already_visited[i]:
                yield from self._edges_from_vertex(i, already_visited)
        yield v


# g = Graph()
# g.add_edge(5, 2)
# g.add_edge(5, 0)
# g.add_edge(4, 0)
# g.add_edge(4, 1)
# g.add_edge(2, 3)
# g.add_edge(3, 1)
# assert g.topological_sort() == [5, 4, 2, 3, 1, 0]
#
# print(list(g.edges()))

