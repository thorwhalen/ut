

from sklearn.model_selection import BaseCrossValidator
from collections import defaultdict
from ut.ml.cluster.supervised import _choose_distribution_according_to_weights
from numpy import ones, array
from numpy.random import choice

DEFAULT_PER_UNIQUE_Y_SPLITS = 10


class SupervisedLeaveOneOut(BaseCrossValidator):
    """Base class for all cross-validators

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def __init__(self, n_splits=None, min_n_samples_per_unik_y=None):
        super(self.__class__, self).__init__()
        self.n_splits = n_splits
        if min_n_samples_per_unik_y is None:
            min_n_samples_per_unik_y = 1
        self.min_n_samples_per_unik_y = min_n_samples_per_unik_y
        self.y_count_ = None
        self.y_idx_ = None
        self.n_unik_ys_ = None
        self.n_to_choose_from_y_ = None

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        self._mk_attrs(X=X, y=y, groups=groups)

        for yy, idx in self.y_idx_.items():
            yy_idx = choice(idx, self.n_to_choose_from_y_[yy], replace=False)
            for this_yy_idx in yy_idx:
                yield this_yy_idx


    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        if self.n_splits is None:
            self._mk_attrs()
            per_unique_y_splits = min(DEFAULT_PER_UNIQUE_Y_SPLITS, min(self.y_count_.values()))
            self.n_splits = per_unique_y_splits * len(self.y_count_)
        return self.n_splits


    def _mk_attrs(self, X=None, y=None, groups=None):
        self.y_idx_ = defaultdict(list)
        list(map(lambda i, y_item: self.y_idx_[y_item].append(i), *list(zip(*enumerate(y)))));
        self.y_count_ = {k: len(v) for k, v in self.y_idx_.items()}
        weights = list(self.y_count_.values())
        self.min_n_samples_per_unik_y = min(min(weights), self.min_n_samples_per_unik_y)

        self.n_unik_ys_ = len(weights)
        if self.n_splits is None:
            self.n_splits = self.min_n_samples_per_unik_y * self.n_unik_ys_
        self.n_splits = max(self.n_splits, self.min_n_samples_per_unik_y * len(weights))

        n_to_choose_from_each_unique_y = \
            _choose_distribution_according_to_weights(array(weights) - self.min_n_samples_per_unik_y,
                                                      self.n_splits - self.min_n_samples_per_unik_y * self.n_unik_ys_) \
            + self.min_n_samples_per_unik_y * ones(self.n_unik_ys_)
        self.n_to_choose_from_y_ = {k: int(v) for k, v in zip(list(self.y_count_.keys()), n_to_choose_from_each_unique_y)}
