

__author__ = 'thor'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
# from ufunc import sqrt, arctan2
from itertools import cycle, islice


def _nstd_for_confidence_level(confidence_level):
    return -scipy.stats.norm.ppf((1 - confidence_level) / 2)


def get_gaussian_ellipse_artist(mean, cov, confidence_level=0.95, color="red", fill=False, linewidth=3, alpha=None,
                                **kwargs):
    """
    Returns an ellipse artist for nstd times the standard deviation of this
    Gaussian, specified by mean and covariance
    """
    nstd = _nstd_for_confidence_level(confidence_level)
    # compute eigenvalues (ordered)
    vals, vecs = np.linalg.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    alpha = alpha or 0.5 if fill else 1

    e = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=theta, \
                            color=color, fill=fill, linewidth=linewidth, alpha=alpha, **kwargs)

    return e


def make_gmm_ellipses(gmm, ax=None, color_palette='rgbcmk', **kwargs):
    if ax is None:
        ax = plt.gca()

    color_iter = cycle(color_palette)
    for i, color in enumerate(islice(color_iter, np.shape(gmm.means_)[0])):
        # ell = mpl.patches.Ellipse(gmm.means_[i, :2], width, height, angle, color=color)
        ell = get_gaussian_ellipse_artist(gmm.means_[i, :2], gmm._get_covars()[i][:2, :2], color=color, **kwargs)
        ell.set_clip_box(ax.bbox)
        #         ell.set_alpha(0.5)
        ax.add_artist(ell)
