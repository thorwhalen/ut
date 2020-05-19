import re
from typing import Optional
import json

from graphviz import Digraph


class DDigraph(Digraph):
    def edges(self, *args, **kwargs):
        super().edges(*args, **kwargs)
        return self


class ModifiedDot:
    class rx:
        lines = re.compile('\n|\r|\n\r|\r\n')
        comments = re.compile('#.+$')
        nout_nin = re.compile(r'(\w+)\W+(\w+)')
        arrow = re.compile(r"\s*->\s*")
        instruction = re.compile(r"(\w+):\s+(.+)")
        node_def = re.compile(r"([\w\s,]+):\s+(.+)")
        wsc = re.compile(r"[\w\s,]+")
        csv = re.compile(r"[\s,]+")
        pref_name_suff = re.compile(r"(\W*)(\w+)(\W*)")

    @staticmethod
    def loose_edges(s):
        return list(
            map(lambda x: x.groups(), filter(None, map(ModifiedDot.rx.nout_nin.search, ModifiedDot.rx.lines.split(s)))))

    # https://www.graphviz.org/doc/info/shapes.html#polygon
    shape_for_chars = {
        ('[', ']'): 'box',
        ('(', ')'): 'circle',
        ('#', '#'): 'box',
        ('/', '/'): 'parallelogram',
        ('<', '>'): 'diamond',
        ('([', '])'): 'cylinder',
        ('[[', ']]'): 'box3d',
        ('((', '))'): 'doublecircle',
        ('/', '\\'): 'triangle',
        ('\\', '/'): 'invtriangle',
        ('|/', '\\|'): 'house',
        ('|\\', '/|'): 'invhouse',
        ('/-', '-\\'): 'trapezium',
        ('-\\', '-/'): 'invtrapezium'
    }

    @staticmethod
    def _modified_dot_gen(s, dflt_node_attr='shape', **dflt_specs):
        csv_items = lambda x: ModifiedDot.rx.csv.split(x.strip())
        pipeline_items = lambda s: list(map(csv_items, s))
        for line in ModifiedDot.rx.lines.split(s):
            statements = ModifiedDot.rx.arrow.split(line)
            if len(statements) > 1:
                pipeline = pipeline_items(statements)
                for nouts, nins in zip(pipeline[:-1], pipeline[1:]):
                    for nout in nouts:
                        for nin in nins:
                            yield 'edge', nout, nin
            else:
                statement = statements[0].strip()
                if statement.startswith('--'):  # it's a special instruction (typically, overriding a default)
                    statement = statement[2:]
                    instruction, specs = ModifiedDot.rx.node_def.search(statement)
                    if instruction == 'dflt_node_attr':
                        dflt_node_attr = specs.strip()
                    else:
                        dflt_specs[instruction] = specs.strip()
                else:  # it's a node definition (or just some stuff to ignore)
                    if statement.startswith('#'):
                        continue  # ignore, it's just a comment
                    g = ModifiedDot.rx.node_def.search(statement)
                    if g is None:
                        continue
                    nodes, specs = g.groups()
                    nodes = csv_items(nodes)
                    if specs.startswith('{'):
                        specs = json.loads(specs)
                    else:
                        specs = {dflt_node_attr: specs}
                    for node in nodes:
                        assert isinstance(specs, dict), \
                            f"specs for {node} be a dict at this point: {specs}"
                        yield 'node', node, dict(dflt_specs, **specs)

    @staticmethod
    def parser(s, **dflt_specs):
        return list(ModifiedDot._modified_dot_gen(s, **dflt_specs))

    @staticmethod
    def interpreter(d, commands, node_shapes, attrs_for_node):
        _edges = list()
        _nodes = {}
        for kind, arg1, arg2 in commands:
            if kind == 'edge':
                from_node, to_node = arg1, arg2
                _edge = list()
                for node in (from_node, to_node):
                    pref, name, suff = ModifiedDot.rx.pref_name_suff.search(node).groups()
                    if ((pref, suff) in node_shapes
                            and name not in _nodes):  # implies that only first formatting (existence of pref and suff) counts
                        _nodes[name] = {'shape': node_shapes[(pref, suff)]}
                        _edge.append(name)
                    else:
                        _edge.append(name)

                _edges.append(_edge)
            elif kind == 'node':
                node, specs = arg1, arg2
                d.node(name=node, **arg2)
        d.edges(_edges)
        for node, attrs in attrs_for_node.items():
            d.node(name=node, **attrs)
        for node, attrs in _nodes.items():
            d.node(name=node, **attrs)
        return d


def dagdisp(commands, node_shapes: Optional[dict] = None,
            attrs_for_node=None,
            minilang=ModifiedDot,
            engine=None, **digraph_kwargs):
    """
    Make a Dag image flexibly.

    Has a mini-language by default (called `ModifiedDot`).

    Example:
    ```
    dagdisp(\"\"\"
        key, wf: circle
        chk: doublecircle
        fv: {"shape": "plaintext", "fontcolor": "blue"}
        key -> wf tag
        wf -> [chunker] -> chk -> /featurizer/ -> fv
        fv tag -> ([model])
        \"\"\"
        )
    ```

    With ModifiedDot you can:

    - specify a bunch of edges at once in a path. For example, a line such as this:
        ```node1 -> node2 -> node3 -> node4```
    will result in these edges
    ```
    [('node1', 'node2'), ('node2', 'node3'), ('node3', 'node4')]
    ```

    - specify 1-to-many, many-to-1 (stars) and many-to-many (bipartite) edges in bulk like this:
    ```
    node1, node2 -> node3, node4, node5  # bipartite graph
    ```

    - use shorthands to shape nodes (see `ModifiedDot.shape_for_chars` for the shape minilanguage, and
        know that you can specify your own additions/modifications)

    - specify node properties in bulk like this:
    ```
    node1, node2, node3 : node_attrs
    ```

    - specify defaults dynamically, within the statements:
    ```
    --fillcolor: red
    --shape: square
    ...
    ```

    :param commands: The commands (edge and node specifications)
    :param node_shapes: Extra tuple-to-shape mappings.
        Used to add to, or override the existing defaults (see them here: `dagdisp.shape_for_chars`).
        This dict constitutes the mini-language used to give shapes to nodes on the fly.
    :param attrs_for_node: See https://www.graphviz.org/doc/info/attrs.html
    :param minilang: Object that populates the graph. Needs a parser and an interpreter method.
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
    :param digraph_kwargs: Other kwargs for graphviz.Digraph(**kwargs)
    :return:
    """
    attrs_for_node = attrs_for_node or {}
    if node_shapes is False:
        node_shapes = {}
    else:
        if node_shapes is True:
            node_shapes = {}
        node_shapes = dict(ModifiedDot.shape_for_chars, **(node_shapes or {}))
    if isinstance(commands, str):
        commands = minilang.parser(commands)
    d = Digraph(engine=engine, **digraph_kwargs)

    minilang.interpreter(d, commands, node_shapes, attrs_for_node)

    return d


dagdisp.shape_for_chars = ModifiedDot.shape_for_chars
