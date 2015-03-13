__author__ = 'thor'

from numpy import *
import numpy as np
import time
import sklearn as sk
import scipy.interpolate as interpolate

from ut.stats.classification.bin.base import BinaryClassifierBase2D
from ut.stats.util import binomial_probs_to_multinomial_probs


class BinaryClassificationByInterpolatedProbabilities(BinaryClassifierBase2D):
    """
    This is a BinaryClassifierBase2D that estimates probabilities by interpolation.
    The fit function finds n_clusters clusters of the x data and assigns to each cluster center the mean of the ys of
    the n_neighbors nearest neighbors of the center.
    The ys of every other point of the x space are then estimated by interpolating over these clusters centers.
    """
    def __init__(self, n_clusters=500, n_neighbors=3000, interpolator='cubic', **kwargs):
        super(BinaryClassificationByInterpolatedProbabilities, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.interpolator = interpolator
        self.clus = []
        self.cluster_x = None
        self.cluster_y = None
        self.iterpol = None
        self.nnb_iterpol = None

    def fit(self, x, y):
        t0 = time.time()

        assert set(y.flatten()) == set([0, 1]), "y data (target data) needs to have only 0s and 1s"

        # determine the clusters
        self.clus = sk.cluster.MiniBatchKMeans(n_clusters=self.n_clusters)
        self.clus.fit(x)

        # determine the nearest neighbor for each data point
        nns = sk.neighbors.NearestNeighbors(n_neighbors=self.n_neighbors)
        nns.fit(x)

        neighbor_dist, neighbor_idx = nns.kneighbors(self.clus.cluster_centers_, n_neighbors=self.n_neighbors)

        # compute the cluster means
        self.cluster_x = self.clus.cluster_centers_
        self.cluster_y = array(map(lambda i: nanmean(y[neighbor_idx[i, :]]), xrange(shape(self.cluster_x)[0])))

        # make the interpolator
        if self.interpolator == 'linear':
            self.iterpol = interpolate.LinearNDInterpolator(self.cluster_x, self.cluster_y, fill_value=nan)
        else:
            self.iterpol = interpolate.CloughTocher2DInterpolator(self.cluster_x, self.cluster_y, fill_value=nan)
        self.nnb_iterpol = interpolate.NearestNDInterpolator(self.cluster_x, self.cluster_y)

        print "fit elapsed time: %.02f minutes" % ((time.time() - t0) / 60.)

    def predict_proba(self, x):
        iterpolations = self.iterpol(x)
        lidx = np.isnan(iterpolations)
        iterpolations[lidx] = self.nnb_iterpol(x)[lidx]
        iterpolations[iterpolations < 0] = 0.0  # cubic interpolation might create negatives, so we set these to 0
        return binomial_probs_to_multinomial_probs(iterpolations)
        # if only_event_probs:
        #     return iterpolations
        # else:
        #     return binomial_probs_to_multinomial_probs(iterpolations)


