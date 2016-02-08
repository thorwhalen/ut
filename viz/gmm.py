from __future__ import division

__author__ = 'thor'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import sin, cos
from itertools import cycle


def make_gmm_ellipses(gmm, ax=None, color_palette='rgbcmk'):
    if ax is None:
        ax = plt.gca()
    x_lims = [0, 0]
    y_lims = [0, 0]

    color_iter = cycle(color_palette)
    for n, color in enumerate(color_iter):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        # angle += 180
        width = v[0]
        height = v[1]

        ell = mpl.patches.Ellipse(gmm.means_[n, :2], width, height, angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

        x = gmm.means_[n, :2][0]
        x_proj = cos(angle) * width
        this_x_lims = sorted([x + x_proj, x - x_proj])
        print(this_x_lims)
        x_lims[0] = min(x_lims[0], this_x_lims[0])
        x_lims[1] = max(x_lims[1], this_x_lims[1])

        y = gmm.means_[n, :2][1]
        y_proj = sin(angle) * height
        this_y_lims = sorted([y + y_proj, y - y_proj])
        y_lims[0] = min(y_lims[0], this_y_lims[0])
        y_lims[1] = max(y_lims[1], this_y_lims[1])

    return x_lims, y_lims
