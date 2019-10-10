

from numpy import cumsum, insert


def running_mean(arr, chk_size):
    c = cumsum(insert(arr, 0, [0]))
    return (c[chk_size:] - c[:-chk_size]) / chk_size
