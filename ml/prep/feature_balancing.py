__author__ = 'thor'

import pandas as pd
from numpy import array, random


def label_balanced_subset(data, label, random_seed=None):

    if not isinstance(data, pd.DataFrame) or label not in data.columns:
        # ... then assume data is the X and label is the y arrays of the supervised learning setup
        return_arrays = True
        dg = pd.concat([pd.DataFrame(data), pd.DataFrame({'label': label})], axis=1).groupby('label')
    else:
        return_arrays = False
        # ... then assume data contains both explanatory and label (targets of classification) data
        dg = data.groupby(label)

    min_count = min(dg.size())

    def get_subset_data_idx(x, random_seed):
        if random_seed == -1:
            return slice(0, min_count)
        else:
            random.seed(random_seed)
            return random.choice(a=len(x), size=min_count, replace=False)

    subset_data = pd.concat([x[1].iloc[get_subset_data_idx(x[1], random_seed)] for x in dg], axis=0)

    if return_arrays:
        y = array(subset_data['label'])
        subset_data.drop('label', axis=1, inplace=True)
        return subset_data.as_matrix(), y
    else:
        return subset_data


