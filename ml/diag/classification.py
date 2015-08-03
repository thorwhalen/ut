__author__ = 'thor'

import pandas as pd


def pred_truth_counts(truth, pred, stat='count'):
    d = pd.DataFrame({'pred': pred, 'truth': truth})[['pred', 'truth']].groupby(['pred', 'truth']).size()
    if stat == 'count':
        d.name = 'count'
    elif stat == 'perc':
        d = 100 * d / sum(d)
        d.name = 'perc'
    elif stat == 'ratio':
        d = d / sum(d)
        d.name = 'ratio'
    elif stat == 'pred':
        d = d / d.groupby(level='pred').sum()
        d = d.reorder_levels(['pred', 'truth'])\
            .sort(inplace=False, ascending=False)\
            .sortlevel('pred', sort_remaining=False)
        d.name = 'P(truth|pred)'
    elif stat == 'truth':
        d = d / d.groupby(level='truth').sum()
        d = d.reorder_levels(['truth', 'pred'])\
            .sort(inplace=False, ascending=False)\
            .sortlevel('truth', sort_remaining=False)
        d.name = 'P(pred|truth)'
    else:
        raise ValueError("Unknown stat (must be count, perc, pred, or truth")
    return d