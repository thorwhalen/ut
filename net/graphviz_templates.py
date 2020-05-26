import graphviz


def graphviz_attrs(**attrs):
    """
    >>> graphviz_attrs(resolution=23, label="My name")
    '[resolution=23, label="My name"]'
    """

    def gen():
        for k, v in attrs.items():
            if isinstance(v, str):
                yield f'{k}="{v}"'
            else:
                yield f'{k}={v}'

    return '[' + ', '.join(gen()) + ']'


class graph_template:
    def __init__(self, name_attrs_str="[shape=rectangle style=filled color=lightgrey]"):
        self.name_attrs_str = name_attrs_str

    def one_to_one(self, name='one_to_one', in_='in', out_='out', label='1-to-1'):
        return f"""
    subgraph cluster_{name} {{
        label = "{label}";
        {name}_in -> {name} -> {name}_out;
        {name}_in [label="{in_}"]
        {name}_out [label="{out_}"]
        {name} {self.name_attrs_str}
    }}"""

    def one_to_many(self, name='one_to_many', in_='in', out_1='out_1', out_n='out_n', label='1-to-many'):
        return f"""
    subgraph cluster_{name} {{
        label = "{label}";
        {name}_in -> {name} -> {name}_out_1, {name}_out_n;
        {name}_in [label="{in_}"]
        {name}_out_1 [label="{out_1}"]
        {name}_out_n [label="{out_n}"]
        {name} {self.name_attrs_str}
    }}"""

    def many_to_one(self, name='many_to_one', in_1='in_1', in_n='in_n', out_='out', label='many-to-1'):
        return f"""
    subgraph cluster_{name} {{
        label = "many-to-1";
        {name}_in_1, {name}_in_n -> {name} -> {name}_out;
        {name}_in_1 [label="{in_1}"]
        {name}_in_n [label="{in_n}"]
        {name}_out [label="{out_}"]
        {name} {self.name_attrs_str}
    }}"""


dflt_graph_template = graph_template()


def mk_graph_source(graph_template=dflt_graph_template, attrs=None, **name_kind):
    attrs = attrs or {}
    template = (
            "digraph G {{\n"
            + "graph {}\n".format(graphviz_attrs(**attrs.get('graph', {})))
            + "node {}\n".format(graphviz_attrs(**attrs.get('node', {})))
            + "edge {}\n".format(graphviz_attrs(**attrs.get('edge', {})))
            + "{}".format('\t\t{}\n' * len(name_kind))
            + "}}")

    def format_specs():
        for name, kind in name_kind.items():
            if isinstance(kind, dict):
                kind = dict(**kind)  # make a copy
                method_name = kind.pop('kind')
                method_kwargs = dict(name=name, **kind)
                method = getattr(graph_template, method_name)
                yield method(**method_kwargs)
            else:
                if isinstance(kind, str):
                    method_name = kind
                    method_args = (name,)
                elif isinstance(kind, tuple):
                    method_name, *method_args = kind
                else:
                    raise TypeError("Can't recognize this kind of kind: {kind}")
                method = getattr(graph_template, method_name)
                yield method(*method_args)

    return template.format(*list(format_specs()))


def mk_graph(graph_template=dflt_graph_template, **name_kind):
    return graphviz.Source(mk_graph_source(graph_template=graph_template, **name_kind))


if __name__ == "__main__":
    """You really need to use this code in a notebook, or capture """
    import graphviz

    gt = graph_template()

    graph = [
        mk_graph(gt, function='one_to_one', wf_source='one_to_many'),

        mk_graph(gt, **{k: 'one_to_many' for k in ('generator', 'wf_source', 'chunker')}),

        mk_graph(gt,
                 generator=dict(kind="one_to_many"),
                 wf_source=dict(kind="one_to_many", in_="source", out_1='wf_1', out_n='wf_n'),
                 chunker=dict(kind="one_to_many", in_="wf", out_1='chk_1', out_n='chk_n'),
                 ),

        mk_graph(gt,
                 function=dict(kind="one_to_one"),
                 featurizer=dict(kind="one_to_one", in_="chk", out_='fv'),
                 quantizer=dict(kind="one_to_one", in_="fv", out_='snip'),
                 snip_stats=dict(kind="one_to_one", in_="snip", out_='stats'),
                 ),

        mk_graph(gt,
                 aggregator=dict(kind="many_to_one"),
                 model_output=dict(kind="many_to_one", in_1="stats_1", in_n="stats_n", out_='detection_info'),
                 ),

        mk_graph(attrs={'graph': dict(size="9,9!"), 'node': dict(size=3)},
                 **{k: 'one_to_many' for k in ('wf_source', 'chunker')}),

        mk_graph(
            attrs={'graph': dict(size="6,6!", fontsize=20),
                   'node': dict(shape="circle", style="filled", color="black",
                                fixedsize="true", width="1.1", height="1.1", fontsize=15)},
            **{k: 'one_to_many' for k in ('wf_source', 'chunker')}),
    ]

    import os

    graph[3].render(filename=os.path.expanduser('~/example_graph.pdf'))
