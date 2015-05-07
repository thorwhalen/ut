__author__ = 'thor'

import pandas as pd


def pred_truth_counts(truth, pred, stat='count'):
    d = pd.DataFrame({'pred': pred, 'truth': truth})[['pred', 'truth']].groupby(['pred', 'truth']).size()
    if stat == 'count':
        return d
    elif stat == 'perc':
        return 100 * d / sum(d)
    elif stat == 'ratio':
        return d / sum(d)
    elif stat == 'pred':
        d = d / d.groupby(level='pred').sum()
        return d.reorder_levels(['pred', 'truth'])\
            .sort(inplace=False, ascending=False)\
            .sortlevel('pred', sort_remaining=False)
    elif stat == 'truth':
        d = d / d.groupby(level='truth').sum()
        return d.reorder_levels(['truth', 'pred'])\
            .sort(inplace=False, ascending=False)\
            .sortlevel('truth', sort_remaining=False)
    else:
        raise ValueError("Unknown stat (must be count, perc, pred, or truth")