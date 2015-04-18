__author__ = 'thor'

import pandas as pd
from numpy import array


def label_balanced_subset(data, label):

    if not isinstance(data, pd.DataFrame) or label not in data.columns:
        # ... then assume data is the X and label is the y arrays of the supervised learning setup
        return_arrays = True
        dg = pd.concat([pd.DataFrame(data), pd.DataFrame({'label': label})], axis=1).groupby('label')
    else:
        return_arrays = False
        # ... then assume data contains both explanatory and label (targets of classification) data
        dg = data.groupby(label)
    min_count = min(dg.size())
    subset_data = pd.concat([x[1].iloc[:min_count] for x in dg], axis=0)
    if return_arrays:
        y = array(subset_data['label'])
        subset_data.drop('label', axis=1, inplace=True)
        return subset_data.as_matrix(), y
    else:
        return subset_data


