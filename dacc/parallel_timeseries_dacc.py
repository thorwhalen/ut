__author__ = 'thor'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from ut.pplot.to import simple_plotly


class ParallelTimeSeriesDacc(object):

    def __init__(self, data_source, date_var, index_var, ts_vars_name='vars', **kwargs):
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source
        elif isinstance(data_source, str):
            if data_source == 'elasticsearch':
                from ut.dacc.es.com import ElasticCom

                es_kwargs = kwargs.get('es_kwargs', {})
                if 'index' in list(kwargs.keys()):
                    es_kwargs['index'] = kwargs.pop('index')
                if 'data_type' in list(kwargs.keys()):
                    es_kwargs['data_type'] = kwargs.pop('data_type')
                ec = ElasticCom(**es_kwargs)

                search_kwargs = kwargs.get('search_kwargs', {})
                search_kwargs = dict({'_id': False}, **search_kwargs)
                exclude_fields = search_kwargs.pop('exclude_fields', [])
                self.df = ec.search_and_export_to_df(exclude_fields=exclude_fields, **search_kwargs)

            else:
                raise NotImplementedError("Unrecognized data_source: {}".format(data_source))
        else:
            raise NotImplementedError("Unrecognized data_source type: {}".format(type(data_source)))

        assert set([date_var, index_var]).issubset(self.df.columns), \
            "Both {} and {} must be columns of the data".format(date_var, index_var)
        self.date_var = date_var
        self.index_var = index_var
        self.ts_vars_name = ts_vars_name
        self.var_names = [x for x in self.df.columns if x not in [self.date_var, self.index_var]]
        self.df.columns.set_names([self.ts_vars_name], inplace=True)

        # pivoting data
        original_length = len(self.df)
        self.df.drop_duplicates(subset=[self.index_var, self.date_var], inplace=True)
        if len(self.df) != original_length:
            raise RuntimeWarning("There are duplicate ({},{}), so I'm deleting offending records"
                                 .format(self.index_var, self.date_var))
            self.df = self.df[~self.df[self.date_var].notnull()]
            raise AssertionError("There are duplicate ({},{}), so I can't pivot the data"
                                 .format(self.index_var, self.date_var))
        self.df = self.df.pivot(index=self.date_var, columns=self.index_var)
        self.df.sort_index(inplace=True)

    def vars_list(self, df=None):
        if df is None:
            df = self.df
        return np.unique(df.columns.get_level_values(level=0))

    def indices_list(self, df=None):
        if df is None:
            df = self.df
        return np.unique(df.columns.get_level_values(level=1))


    @staticmethod
    def drop_columns_with_insufficient_dates(d, min_num_of_dates):
        """
        Drop columns that don't have a minimum number of non-NaN dates
        """
        print(("original shape: {}".format(d.shape)))
        num_of_dates = (~d.isnull()).sum()
        num_of_dates = num_of_dates[num_of_dates > min_num_of_dates].sort(inplace=False, ascending=False)
        d = d[num_of_dates.index.values].dropna(how='all')
        print(("shape with at least {} dates: {}".format(min_num_of_dates, d.shape)))
        return d

    @staticmethod
    def latest_full_shape_choices(d):
        """
        Get a table describing the shapes of all
        """
        shape_choices = list()
        for i in range(1, len(d)):
            this_shape = d.iloc[-i:].dropna(axis=1).shape
            shape_choices.append({'i': i, 'rows': this_shape[0], 'cols': this_shape[1]})
        shape_choices = pd.DataFrame(shape_choices).set_index('i')
        shape_choices['pts'] = shape_choices['rows'] * shape_choices['cols']
        return shape_choices

    def print_percentages_of_xvar_more_than_yvar(self, xvar, yvar, min_y=0, df=None):
        if df is None:
            df = self.df.stack(self.index_var)
        t = df[[xvar, yvar]].dropna()
        t = t[t[yvar] >= min_y]
        n_xvar_more_than_yvar = sum(t[xvar] > t[yvar])
        print(("{:.2f}% ({}/{}) of '{}' > '{}'".format(100 * n_xvar_more_than_yvar / float(len(t)),
                                                      n_xvar_more_than_yvar, len(t),
                                                      xvar, yvar)))

    def plot_time_series(self, d, title=None, y_labels=None,
                         width_factor=2, length=18, only_first_non_null=True, with_plotly=False):
        # idSite = 349

        if isinstance(d, tuple):
            d = self.df.loc[:, d]
        if only_first_non_null:
            lidx = np.any(d.notnull(), axis=1)
            d = d.iloc[lidx]

        default_title, default_y_labels = _choose_title_and_y_label(d)
        title = title or default_title
        y_labels = y_labels or default_y_labels

        last_ax = None
        n = len(d.columns)
        fig = plt.figure(figsize=(length, min(n, 50) * width_factor))
        for i, tt in enumerate(d.items()):
            plt.subplot(n, 1, i + 1)
            tt[1].index = tt[1].index.map(pd.to_datetime)

            tt[1].plot(sharex=last_ax)
            ax = plt.gca()

            if title == 'y_labels':
                ax.set_title(y_labels[i])
            else:
                if i == 0:
                    ax.set_title(title)
                if isinstance(y_labels[i], str):
                    plt.ylabel(y_labels[i].replace('_', '\n'))
                else:
                    plt.ylabel(y_labels[i])
                ax.yaxis.set_label_position("right")

            if i + 1 < n:
                plt.xlabel('')
            last_ax = ax

        if with_plotly:
            return simple_plotly(fig)

    def get_plotly_url(self, plotly_obj):
        if hasattr(plotly_obj, 'embed_code'):
            return re.compile('src="([^"]*)"').search(plotly_obj.embed_code).group(1)


def _choose_title_and_y_label(d):
    col_vals = d.columns.values
    try:
        level_1_vals, level_2_vals = list(zip(*col_vals))
        if len(np.unique(level_1_vals)) == 1:
            return level_1_vals[1], level_2_vals
        elif len(np.unique(level_2_vals)) == 1:
            return level_2_vals[0], level_1_vals
        else:
            return " & ".join(d.columns.names), col_vals
    except TypeError:
        return " & ".join(d.columns.names), col_vals
