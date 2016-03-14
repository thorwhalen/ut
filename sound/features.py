__author__ = 'thor'

from numpy import *


def crescendoness(wf, sr, averaging_window_seconds=0.5):
    """
    Computes a score describing how much the sound's intensity is monotone increasing (if crescendoness is positive)
    or decreasing (if crescendoness is negative).

    The moving sum of the differential of the absolute value of the wave form is computed
    with a window of averaging_window_seconds seconds, then the log2 of the proportion of the positive and negativ
    is returned as the score.

    Basically, the score indicates how many of these averaging_window_seconds sized windows have a sound that is
    increasing in "volume" compared to how many are decreasing.
    """
    y = _moving_sum(abs(wf), round(sr * averaging_window_seconds))
    w = unique(diff(y) > 0, return_counts=True)
    return log2(w[1][w[0]] / float(w[1][~w[0]]))[0]


def _moving_sum(a, window_size):
    ret = cumsum(a, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:]

