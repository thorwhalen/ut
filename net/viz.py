"""
Making diagrams easily.

Need to have graphviz (pip install graphviz), but also need the backend of this python binder:
Mac: brew install graphviz
Linux: sudo apt-get install graphviz
Windows: google it
"""

import re
from typing import Optional
import json
from collections import defaultdict
from functools import wraps
from graphviz import Digraph, Source

from types import MethodType

# Note: Not used anywhere in the module anymore, but was
"""Get a `re.Pattern` instance (as given by re.compile()) with control over defaults of it's methods.
Useful to reduce if/else boilerplate when handling the output of search functions (match, search, etc.)

See [regex_search_hack.md](https://gist.github.com/thorwhalen/6c913e9be35873cea6efaf6b962fde07) for more explanatoins of the 
use case.

Example;
>>> dflt_result = type('dflt_search_result', (), {'groupdict': lambda x: {}})()
>>> p = re_compile('.*(?P<president>obama|bush|clinton)', search=dflt_result, match=dflt_result)
>>>
>>> p.search('I am beating around the bush, am I?').groupdict().get('president', 'Not found')
'bush'
>>>
>>> # if not match is found, will return 'Not found', as requested
>>> p.search('This does not contain a president').groupdict().get('president', 'Not found')
'Not found'
>>>
>>> # see that other non-wrapped re.Pattern methods still work
>>> p.findall('I am beating arcached_keysound the bush, am I?')
['bush']
"""

import re
from functools import wraps


def add_dflt(func, dflt_if_none):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is not None:
            return result
        else:
            return dflt_if_none

    return wrapped_func


def re_compile(pattern, flags=0, **dflt_if_none):
    """Get a `re.Pattern` instance (as given by re.compile()) with control over defaults of it's methods.
    Useful to reduce if/else boilerplate when handling the output of search functions (match, search, etc.)

    Example;
    >>> dflt_result = type('dflt_search_result', (), {'groupdict': lambda x: {}})()
    >>> p = re_compile('.*(?P<president>obama|bush|clinton)', search=dflt_result, match=dflt_result)
    >>>
    >>> p.search('I am beating around the bush, am I?').groupdict().get('president', 'Not found')
    'bush'
    >>> p.match('I am beating around the bush, am I?').groupdict().get('president', 'Not found')
    'bush'
    >>>
    >>> # if not match is found, will return 'Not found', as requested
    >>> p.search('This does not contain a president').groupdict().get('president', 'Not found')
    'Not found'
    >>>
    >>> # see that other non-wrapped re.Pattern methods still work
    >>> p.findall('I am beating around the bush, am I?')
    ['bush']
    """
    compiled_regex = re.compile(pattern, flags=flags)
    intercepted_names = set(dflt_if_none)

    my_regex_compilation = type('MyRegexCompilation', (object,), {})()

    for _name, _dflt in dflt_if_none.items():
        setattr(my_regex_compilation, _name, add_dflt(getattr(compiled_regex, _name), _dflt))
    for _name in filter(lambda x: not x.startswith('__') and x not in intercepted_names,
                        dir(compiled_regex)):
        setattr(my_regex_compilation, _name, getattr(compiled_regex, _name))

    return my_regex_compilation


class rx:
    name = re.compile('^:(\w+)')
    lines = re.compile('\n|\r|\n\r|\r\n')
    comments = re.compile('#.+$')
    non_space = re.compile('\S')
    nout_nin = re.compile(r'(\w+)\W+(\w+)')
    arrow = re.compile(r"\s*->\s*")
    instruction = re.compile(r"(\w+):\s+(.+)")
    node_def = re.compile(r"([\w\s,]+):\s+(.+)")
    wsc = re.compile(r"[\w\s,]+")
    csv = re.compile(r"[\s,]+")
    pref_name_suff = re.compile(r"(\W*)(\w+)(\W*)")


class DDigraph(Digraph):
    @wraps(Digraph.__init__)
    def __init__(self, *args, **kwargs):
        if args:
            first_arg = args[0]
            if first_arg.startswith(':'):
                lines = rx.lines.split(first_arg)
                first_line = lines[0]
                name = (rx.name.search(first_line).group(1))


class ModifiedDot:
    class rx:
        lines = re.compile('\n|\r|\n\r|\r\n')
        comments = re.compile('#.+$')
        non_space = re.compile('\S')
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
        (']', '['): 'square',
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
            line = ModifiedDot.rx.comments.sub('', line)
            statements = ModifiedDot.rx.arrow.split(line)
            if len(statements) > 1:
                pipeline = pipeline_items(statements)
                for nouts, nins in zip(pipeline[:-1], pipeline[1:]):
                    for nout in nouts:
                        for nin in nins:
                            yield 'edge', nout, nin
            else:
                statement = statements[0].strip()
                if ':' in statement:
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
                elif ModifiedDot.rx.non_space.search(statement):
                    yield 'source', statement, None

    @staticmethod
    def parser(s, **dflt_specs):
        return list(ModifiedDot._modified_dot_gen(s, **dflt_specs))

    @staticmethod
    def interpreter(commands, node_shapes, attrs_for_node, engine, **digraph_kwargs):
        _edges = list()
        _nodes = defaultdict(dict)
        _sources = list()
        for kind, arg1, arg2 in commands:
            if kind == 'edge':
                from_node, to_node = arg1, arg2
                _edge = list()
                for node in (from_node, to_node):
                    pref, name, suff = ModifiedDot.rx.pref_name_suff.search(node).groups()
                    if ((pref, suff) in node_shapes
                            and name not in _nodes):  # implies that only first formatting (existence of pref and suff) counts
                        _nodes[name].update(shape=node_shapes[(pref, suff)])
                        _edge.append(name)
                    else:
                        _edge.append(name)

                _edges.append(_edge)
            elif kind == 'node':
                node, specs = arg1, arg2
                _nodes[node].update(**arg2)
            elif kind == 'source':
                _sources.append(arg1)

        digraph_kwargs['body'] = digraph_kwargs.get('body', []) + _sources
        d = Digraph(engine=engine, **digraph_kwargs)

        d.edges(_edges)
        for node, attrs in attrs_for_node.items():
            d.node(name=node, **attrs)
        for node, attrs in _nodes.items():
            d.node(name=node, **attrs)

        return d


def dgdisp(commands,
           node_shapes: Optional[dict] = None,
           attrs_for_node=None,
           minilang=ModifiedDot,
           engine=None, **digraph_kwargs):
    """
    Make a Dag image flexibly.

    Quick links:
    - attributes: https://www.graphviz.org/doc/info/attrs.html
    - shapes: https://www.graphviz.org/doc/info/shapes.html#polygon

    Has a mini-language by default (called `ModifiedDot`).

    Example:
    ```
    dgdisp(\"\"\"
        key, wf: circle
        chk: doublecircle
        fv: {"shape": "plaintext", "fontcolor": "blue"}
        key -> wf tag
        wf -> [chunker] -> chk -> /featurizer/ -> fv
        fv tag -> ([model])
        \"\"\"
        )
    ```

    ```
    d = dgdisp(\"\"\"
        group_tags, orig_tags -> [mapping] -> tags  # many-to-1 and path (chain) example
        predicted_tags, \\tags/ -> /confusion_matrix/  # you can format the shape of nodes inplace
        predict_proba, tag_list -> [[predict]] -> /predicted_tags\\
        group_tags: {"fillcolor": "red", "fontcolor": "red"}  # you can specify node attributes as json
        orig_tags [fontsize=30 fontcolor=blue]  # you can write graphviz lines as is
        # tag_list [shape=invhouse fontcolor=green]  # you can comment out lines
        \"\"\", format='svg')
        d.render('svg')
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
        Used to add to, or override the existing defaults (see them here: `dgdisp.shape_for_chars`).
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
        node_shapes = dict(minilang.shape_for_chars, **(node_shapes or {}))
    if isinstance(commands, str):
        commands = minilang.parser(commands)

    d = minilang.interpreter(commands, node_shapes, attrs_for_node, engine=engine, **digraph_kwargs)

    return d


dgdisp.shape_for_chars = ModifiedDot.shape_for_chars


@wraps(dgdisp)
def horizontal_dgdisp(*args, **kwargs):
    command, *_args = args
    return dgdisp('rankdir="LR"\n' + command, *_args, **kwargs)


dgdisp.h = horizontal_dgdisp


class Struct:
    def __init__(self, **kwargs):
        for attr, val in kwargs.items():
            setattr(self, attr, val)


dgdisp.engines = Struct(**{x: x for x in ['dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo']})

dagdisp = dgdisp
