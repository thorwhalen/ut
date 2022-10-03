__author__ = 'thor'


def sequential_ratios(d, **kwargs):
    # input processing
    prior_weight = kwargs.get('prior_weight', 1.0)
    specs = kwargs.get('specs', None)
    if specs is None:
        specs = [{'num': d.columns[0], 'denom': d.columns[1]}]
    if isinstance(specs, dict):
        specs = [specs]
    for s in specs:
        if 'prior_weight' not in list(s.keys()):
            s['prior_weight'] = prior_weight
    # computing level stats

    for level in range(len(d.index.levels)):
        pass
