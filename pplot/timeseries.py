from collections import Counter
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import collections as mc

from ut.pplot.date_ticks import str_ticks
from ut.util.utime import utc_ms_to_utc_datetime
from ut.pdict.get import get_value_in_key_path

DFLT_FIG_WIDTH = 16
DFLT_FIG_HEIGHT = 5
DFLT_FIT_HEIGHT_FACTOR = 2.5
DFLT_MAX_FIG_HEIGHT = 100
DFLT_FIGSIZE = (DFLT_FIG_WIDTH, DFLT_FIG_HEIGHT)
DFLT_ALPHA = 0.5

DFLT_TS_FIELD = 'offset_date'
DFLT_BT_FIELD = 'bt'
DFLT_TT_FIELD = 'tt'

DFLT_CHANNEL_FIELD = 'c'
DFLT_TS_VAL_FIELD = 'signal_val'
DFLT_VAL_FIELD = 'v'

ms_seconds = 0.001
inf = float('inf')


class TimeseriesPlot(object):
    def __init__(
        self,
        ts_field=DFLT_TS_FIELD,
        ts_val_field=DFLT_TS_VAL_FIELD,
        fig_width=DFLT_FIG_WIDTH,
        fig_height_factor=DFLT_FIT_HEIGHT_FACTOR,
        max_fig_height=DFLT_MAX_FIG_HEIGHT,
    ):
        self.ts_field = ts_field
        self.ts_val_field = ts_val_field
        self.fig_width = fig_width
        self.fig_height_factor = fig_height_factor
        self.max_fig_height = max_fig_height

    def plot_timeseries(self, timeseries):
        n = len(list(timeseries.keys()))
        min_offset_date = min(
            [min(x[self.ts_field]) for x in list(timeseries.values())]
        )
        max_offset_date = max(
            [max(x[self.ts_field]) for x in list(timeseries.values())]
        )
        min_datetime = utc_ms_to_utc_datetime(min_offset_date)
        max_datetime = utc_ms_to_utc_datetime(max_offset_date)

        fig = plt.figure(
            figsize=(
                self.fig_width,
                min(self.max_fig_height, self.fig_height_factor * n),
            )
        )

        for i, (named_signal, d) in enumerate(iter(timeseries.items()), 1):
            ax = fig.add_subplot(n, 1, i)

            ax.text(
                0.5,
                0.9,
                named_signal,
                horizontalalignment='center',
                transform=ax.transAxes,
            )

            signal_val = d[self.ts_val_field]
            if isinstance(signal_val[0], str):
                sr = pd.Series(Counter(signal_val)).sort_values(ascending=False)
                sr.name = named_signal
                sr.plot(kind='bar', ax=ax)
                ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            else:
                x = list(map(utc_ms_to_utc_datetime, d[self.ts_field]))
                ax.plot(x, signal_val, '-o')
                plt.xlim([min_datetime, max_datetime])

        return fig


class SegmentPlot(object):
    def __init__(
        self,
        bt_field=DFLT_BT_FIELD,
        tt_field=DFLT_TT_FIELD,
        val_field=DFLT_VAL_FIELD,
        figsize=DFLT_FIGSIZE,
        alpha=DFLT_ALPHA,
        fig_width=DFLT_FIG_WIDTH,
        fig_height_factor=DFLT_FIT_HEIGHT_FACTOR,
        max_fig_height=DFLT_MAX_FIG_HEIGHT,
    ):
        self.bt_field = bt_field
        self.tt_field = tt_field
        self.val_field = val_field
        self.figsize = figsize
        self.alpha = alpha
        self.fig_width = fig_width
        self.fig_height_factor = fig_height_factor
        self.max_fig_height = max_fig_height

    def plot_segments_df(self, segments_df, ax=None):
        categorical_data = False
        if isinstance(segments_df.iloc[0][self.val_field], str):
            categorical_data = True
            idx_to_cat = segments_df[self.val_field].unique()
            cat_to_idx = {cat: idx for idx, cat in enumerate(idx_to_cat)}
            segments_df[self.val_field] = list(
                map(cat_to_idx.get, segments_df[self.val_field])
            )
        else:
            idx_to_cat = None

        x = list(
            zip(
                list(zip(segments_df[self.bt_field], segments_df[self.val_field])),
                list(zip(segments_df[self.tt_field], segments_df[self.val_field])),
            )
        )
        lc = mc.LineCollection(x)

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        ax.add_collection(lc)
        ax.autoscale()
        plt.xlabel('Time')

        if self.alpha > 0:
            ax.plot(
                segments_df[self.bt_field],
                segments_df[self.val_field],
                'o',
                alpha=self.alpha,
            )

        if categorical_data:
            plt.yticks(list(range(len(idx_to_cat))), idx_to_cat)

        x_ticks, _ = plt.xticks()
        plt.xticks(x_ticks, str_ticks(x_ticks, ticks_unit=ms_seconds))

        return ax

    def plot_multiple_segments(self, signal_segments_it, n=None):
        if n is None:
            try:
                n = len(signal_segments_it)
            except AttributeError:
                n = signal_segments_it.count()  # like if it's a mongo cursor

        min_t = -inf
        max_t = inf

        fig = plt.figure(
            figsize=(
                self.fig_width,
                min(self.max_fig_height, self.fig_height_factor * n),
            )
        )

        for i, (signal, d) in enumerate(signal_segments_it, 1):
            ax = fig.add_subplot(n, 1, i)
            d = pd.DataFrame(d)
            min_t = min(d[self.bt_field].min(), min_t)
            max_t = max(d[self.tt_field].max(), max_t)

            ax.text(
                0.5, 0.9, signal, horizontalalignment='center', transform=ax.transAxes
            )

            self.plot_segments_df(segments_df=d, ax=ax)

        return fig


class MgSegmentPlot(SegmentPlot):
    def __init__(
        self,
        mgc,
        channel_field=DFLT_CHANNEL_FIELD,
        mg_val_field=DFLT_VAL_FIELD,
        bt_field=DFLT_BT_FIELD,
        tt_field=DFLT_TT_FIELD,
        val_field='mg_val_field_tail',
        figsize=DFLT_FIGSIZE,
        alpha=DFLT_ALPHA,
        fig_width=DFLT_FIG_WIDTH,
        fig_height_factor=DFLT_FIT_HEIGHT_FACTOR,
        max_fig_height=DFLT_MAX_FIG_HEIGHT,
    ):
        self.mgc = mgc
        self.channel_field = channel_field
        self.mg_val_field = mg_val_field
        self.mg_val_key_path = self.val_field.split('.')
        if val_field == 'mg_val_field_tail':
            val_field = self.mg_val_key_path[-1]
        elif val_field == 'mg_val_field_head':
            val_field = self.mg_val_key_path[0]
        super(MgSegmentPlot, self).__init__(
            bt_field=bt_field,
            tt_field=tt_field,
            val_field=val_field,
            figsize=figsize,
            alpha=alpha,
            fig_width=fig_width,
            fig_height_factor=fig_height_factor,
            max_fig_height=max_fig_height,
        )

    def segments_df_from_mgc(self, mgc, channel):
        it = mgc.find(
            {self.channel_field: channel},
            fields={
                '_id': 0,
                self.mg_val_key_path: 1,
                self.bt_field: 1,
                self.tt_field: 1,
            },
        )
        return pd.DataFrame(
            list(
                zip(
                    *[
                        (
                            get_value_in_key_path(doc, self.mg_val_key_path),
                            doc[self.bt_field],
                            doc[self.tt_field],
                        )
                        for doc in it
                    ]
                )
            ),
            index=[self.val_field, self.bt_field, self.tt_field],
        ).T

    def plot_segments_for_signal_from_mgc(self, mgc, channel):
        segments_df = self.segments_df_from_mgc(mgc, channel)
        ax = self.plot_segments_df(segments_df)
        plt.ylabel(channel)
        return ax
