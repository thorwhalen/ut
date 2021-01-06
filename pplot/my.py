"""What matplotlib has already -- but with the interface I'd like"""

from numpy import *
import matplotlib.pylab as plt


def vlines(x, ymin=0, ymax=None, marker='o', marker_kwargs=None,
           colors='k', linestyles='solid', label='', hold=None, data=None, **kwargs):
    if ymax is None:
        ymax = x
        x = arange(len(ymax))

        if ymax is None:
            raise ValueError("Need to specify ymax")

    if marker is not None:
        if marker_kwargs is None:
            marker_kwargs = {}
        plt.plot(x, ymax, marker, **marker_kwargs)

    return plt.vlines(x, ymin=ymin, ymax=ymax,
                      colors=colors, linestyles=linestyles, label=label, hold=hold, data=data, **kwargs)
