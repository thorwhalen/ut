__author__ = 'thor'

import numpy as np


def insert_nans_in_x_and_y_when_there_is_a_gap_in_x(x, y, gap_thresh):
    assert len(x) == len(y), "x and y must be the same length"
    assert sorted(x) == x, "x must be sorted"
    new_x = np.array([])
    new_y = np.array([])
    x_ref = x[0]
    for i in range(len(x)):
        gap = x[i] - x_ref
        if gap <= gap_thresh:
            new_x = np.append(new_x, x[i])
            new_y = np.append(new_y, y[i])
        else:
            new_x = np.append(new_x, [np.nan, x[i]])
            new_y = np.append(new_y, [np.nan, y[i]])
        x_ref = x[i]
    return new_x, new_y
