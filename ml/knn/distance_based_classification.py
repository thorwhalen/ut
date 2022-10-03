__author__ = 'thor'


import pandas as pd


class DistanceClf(object):
    def __init__(
        self,
        dist_df,  # a df (or ndarray) indexed (rows and columns) by record ids and where df.loc[i,j]=dist(i,j)
        labels,  # an array, dict or series mapping record ids to classification labels
    ):
        if isinstance(labels, dict):
            labels = pd.Series(labels)
        elif not isinstance(labels, pd.Series):
            labels = {i: x for i, x in enumerate(labels)}
        self.labels = labels

        if not isinstance(
            dist_df, pd.DataFrame
        ):  # if dist_df is not a dataframe, assume it's an ndarray, and make a df out of it
            self.dist_df = pd.DataFrame(
                data=dist_df, index=self.labels.index, columns=self.labels.index
            )
        else:
            self.dist_df = dist_df
        original_dist_df_shape = self.dist_df.shape

        self.dist_df = self.dist_df.loc[self.labels.index, self.labels.index]
        if self.dist_df.shape != original_dist_df_shape:
            print(
                "Indices of labels and dist_df weren't aligned. Kept only records present in labels"
            )
            print(
                (
                    '--> original_dist_df_shape={}, new_dist_df_shape={}'.format(
                        original_dist_df_shape, self.dist_df.shape
                    )
                )
            )

    def sort_records(self, record_order=None):
        if record_order is None:
            self.dist_df.loc[self.labels.index, self.labels.index]
        else:
            self.dist_df = self.dist_df.loc[record_order, record_order]
            self.labels = self.labels.loc[record_order]

    def assert_record_alignment(self):
        """
        Assert that records are all in the same order (in self.labels, and self.dist_df indices and columns)
        """
        assert len(self.dist_df.index.values) == len(
            self.dist_df.columns
        ), "dist_df.index and dist_df.columns don't have the same number of elements"
        assert all(
            self.dist_df.index.values == self.dist_df.columns
        ), 'dist_df.index and dist_df.columns are not aligned'
        if len(self.labels) > 0:
            assert len(self.dist_df.index.values) == len(
                self.labels.index.values
            ), "labels.index and dist_df.index don't have the same number of elementss"
            assert all(
                self.dist_df.index.values == self.labels.index.values
            ), 'labels.index and dist_df.index are not aligned'
