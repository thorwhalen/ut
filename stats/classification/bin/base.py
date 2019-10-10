__author__ = 'thor'


from numpy import *
import matplotlib.pyplot as plt

from ut.stats.util import df_picker_data_prep


class BinaryClassifierBase2D(object):
    """
    Base class for binary classification with 2D explanatory variable space.
    It follows the pattern of sklearn classifiers, with a fit, predict_proba, and predict methods.
    In addition,
        * the predict_proba method contains a only_event_probs flag that can be set to True to only return
            success probabilities instead of pairs of multinomial probabilities.
        * data_prep attribute can be specified to facilitate input of data from a raw form to matrices that are used in
            fit and estimation
        * Various visualization methods are provided
        * The default probability estimation model returns the mean event rate.
    """
    def __init__(self, x_name=['x0', 'x1'], y_name='event', y_prob_name='event_probability', data_prep=None):
        self.x_name = x_name
        self.y_name = y_name
        self.y_prob_name = y_prob_name
        self.mean_y = None
        self.x_range = None
        # default data prep function just picks columns from input
        self.data_prep = data_prep or df_picker_data_prep(self.x_name, self.y_name)

    # the three following methods (fit, predict_proba, and predict) are methods that are found in many sklearn models
    def fit(self, x, y):
        assert set(y[:, 0]) == set([0, 1]), "y data (target data) needs to have only 0s and 1s"
        self.mean_y = nanmean(y)
        self.x_range = [(nanmin(x[:, 0]), nanmax(x[:, 0])), (nanmin(x[:, 1]), nanmax(x[:, 1]))]

    def predict_proba(self, x):
        return tile([1 - self.mean_y, self.mean_y], (shape(x)[0], 1))
        # old version:
        # if only_event_probs:
        #     return repeat(self.mean_y, shape(x)[0])
        # else:
        #     return tile([1 - self.mean_y, self.mean_y], (shape(x)[0], 1))

    def predict(self, x, prob_thresh=0.5):
        return (self.predict_proba(x)[:, 1] > prob_thresh).astype(float)

    # the following are extra methods I wanted to have

    def prob_of_event(self, x):
        probs = self.predict_proba(x)
        return probs[:, 1]
        # old version:
        # return self.predict_proba(x, only_event_probs=True)

    def image(self, point_grid=None, cmap='hot'):
        """
        makes an image of the (2-dimensional) x and predicted y
        Input:
            point_grid:
                a grid constructed with mgrid[...]
                tuple (max_x0, max_x1) specifying the grid mgrid[0:max_x0:100j, 0:max_x1:100j]
                defaults to mgrid[0:max(x[:,0]):100j, 0:max(x[:,1]):100j]
        """
        if point_grid is None:
            point_grid = self.mk_grid('minmax')
        elif isinstance(point_grid, tuple):
            point_grid = self.mk_grid(point_grid)

        n_xgrid = shape(point_grid)[1]
        n_ygrid = shape(point_grid)[2]

        positions = vstack(list(map(ravel, point_grid))).transpose()
        plt.pcolor(positions[:, 0].reshape(n_xgrid, n_ygrid),
                   positions[:, 1].reshape(n_xgrid, n_ygrid),
                   self.predict_proba(positions)[:, 1]
                       .reshape(n_xgrid, n_ygrid), cmap=cmap)
        plt.tight_layout()
        plt.colorbar()
        plt.xlabel(self.x_name[0])
        plt.ylabel(self.x_name[1])
        plt.title(self.y_prob_name)

    def slice_probs(self, x0, x1):
        x = transpose([tile(x0, len(x1)), repeat(x1, len(x0))])
        return self.predict_proba(x)[:, 1]

    @staticmethod
    def _other_dim(dim):
        if dim == 0:
            return 1
        elif dim == 1:
            return 0
        else:
            ValueError("dim should be 0 or 1")

    def plot_slices(self, slice_points, slice_dim=0, line_points=None):
        other_dim = self._other_dim(slice_dim)
        if line_points is None:
            line_points = linspace(self.x_range[other_dim][0], self.x_range[other_dim][0], 100)
        for slice_pt in slice_points:
            if slice_dim == 0:
                plt.plot(line_points, self.slice_probs([slice_pt], line_points), label=slice_pt)
            else:
                plt.plot(line_points, self.slice_probs(line_points, [slice_pt]), label=slice_pt)
        plt.xlabel(self.x_name[other_dim])
        plt.ylabel(self.y_prob_name)
        plt.legend(framealpha=0.5, title=self.x_name[slice_dim])

    def mk_grid(self, method='minmax'):
        if isinstance(method, str):
            if method == 'minmax':
                return mgrid[slice(percentile(self.cluster_X[:, 0], 0), percentile(self.cluster_X[:, 0], 100), 100j),
                            slice(percentile(self.cluster_X[:, 1], 0), percentile(self.cluster_X[:, 1], 100), 100j)]
            elif method == 'alpha':
                return mgrid[linspace(percentile(self.cluster_X[:, 0], 5), percentile(self.cluster_X[:, 0], 95), 100j),
                            linspace(percentile(self.cluster_X[:, 1], 5), percentile(self.cluster_X[:, 1], 95), 100j)]
            else:
                raise ValueError("unrecognized method")
        else:
            return mgrid[slice(0, method[0], 100j), slice(0, method[1], 100j)]
