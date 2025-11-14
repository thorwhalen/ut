__author__ = 'thor'

from wordcloud import WordCloud
import colorsys
import seaborn as sns
from numpy import sqrt, linspace, ceil, where, arange, array, any, floor, ceil, ndarray
from pandas import Series
import matplotlib.pyplot as plt


class TopicExplorer:
    def __init__(
        self,
        url_vectorizer,
        topic_model,
        topic_weight_normalization=None,
        word_preprocessor=None,
        wordcloud_params={
            'ranks_only': True,
            'width': 300,
            'height': 300,
            'margin': 1,
            'background_color': 'black',
        },
        replace_empty_feature_with='EMPTY',
        word_art_params={},
    ):
        self.url_vectorizer = url_vectorizer
        self.feature_names = self.url_vectorizer.get_feature_names()
        if word_preprocessor is None:
            self.word_preprocessor = lambda x: x
        else:
            self.word_preprocessor = word_preprocessor

        # some features might have empty names: Replace them with replace_empty_feature_with
        if replace_empty_feature_with is not None:
            lidx = array(self.feature_names) == ''
            if any(lidx):
                self.feature_names[lidx] = replace_empty_feature_with

        self.topic_model = topic_model
        self.wordcloud_params = wordcloud_params
        self.word_art_params = word_art_params
        self.n_topics = len(self.topic_model.components_)

        topic_components = self.topic_model.components_
        if topic_weight_normalization is not None:
            if isinstance(topic_weight_normalization, str):
                if topic_weight_normalization == 'tf_normal':

                    def topic_weight_normalization(topic_components):
                        topic_components /= topic_components.sum(axis=1)[:, None]
                        topic_components *= 1 / sqrt(
                            (topic_components ** 2).sum(axis=0)
                        )
                        return topic_components

                else:
                    ValueError('Unknown topic_weight_normalization name')

            if callable(topic_weight_normalization):
                topic_components = topic_weight_normalization(topic_components)

        self.topic_word_weights = list()
        for topic_idx, topic in enumerate(topic_components):
            topic_ww = dict()
            for i in topic.argsort():
                topic_ww[self.feature_names[i]] = topic_components[topic_idx, i]
            self.topic_word_weights.append(
                Series(topic_ww).sort_values(ascending=False, inplace=False)
            )

        self.topic_color = ['hsl(0, 100%, 100%)']

        h_list = list(map(int, linspace(0, 360, len(self.topic_model.components_))))[
            :-1
        ]
        for h in h_list:
            self.topic_color.append(f'hsl({h}, 100%, 50%)')

    def topic_weights(self, text_collection):
        if isinstance(text_collection, str):
            urls = [text_collection]
        return self.topic_model.transform(
            self.url_vectorizer.transform(text_collection)
        )

    def topic_word_art(
        self,
        topic_idx=None,
        n_words=20,
        save_file=None,
        color_func=None,
        random_state=1,
        fig_row_size=16,
        **kwargs
    ):
        if topic_idx is None:
            ncols = int(floor(sqrt(self.n_topics)))
            nrows = int(ceil(self.n_topics / float(ncols)))
            ncols_to_nrows_ratio = ncols / nrows
            plt.figure(figsize=(fig_row_size, ncols_to_nrows_ratio * fig_row_size))
            for i in range(self.n_topics):
                plt.subplot(nrows, ncols, i + 1)
                self.topic_word_art(
                    topic_idx=i,
                    n_words=n_words,
                    save_file=save_file,
                    color_func=color_func,
                    random_state=random_state,
                    **kwargs
                )
            plt.gcf().subplots_adjust(wspace=0.1, hspace=0.1)
        # elif isinstance(topic_idx, (list, tuple, ndarray)) and len(topic_idx) == self.n_topics:
        #     ncols = int(floor(sqrt(self.n_topics)))
        #     nrows = int(ceil(self.n_topics / float(ncols)))
        #     ncols_to_nrows_ratio = ncols / nrows
        #     plt.figure(figsize=(fig_row_size, ncols_to_nrows_ratio * fig_row_size))
        #     for i in range(self.n_topics):
        #         plt.subplot(nrows, ncols, i + 1)
        #         self.topic_word_art(topic_idx=i, n_words=n_words, save_file=save_file,
        #                             color_func=color_func, random_state=random_state,
        #                             width=int(self.wordcloud_params['width'] * topic_idx[i]),
        #                             height=int(self.wordcloud_params['height'] * topic_idx[i]))
        #     plt.gcf().subplots_adjust(wspace=.1, hspace=.1)
        else:
            kwargs = dict(self.wordcloud_params, **kwargs)
            if color_func is None:
                color_func = self.word_art_params.get(
                    'color_func', self.topic_color[topic_idx]
                )
            if isinstance(color_func, tuple):
                color_func = 'rgb({}, {}, {})'.format(*list(map(int, color_func)))
            if isinstance(color_func, str):
                color = color_func

                def color_func(
                    word, font_size, position, orientation, random_state=None, **kwargs
                ):
                    return color

            elif not callable(color_func):
                TypeError(f'Unrecognized hsl_color type ()')

            # kwargs = dict(self.word_art_params, **kwargs)
            wc = WordCloud(random_state=random_state, **kwargs)
            wc.fit_words(
                [
                    (self.word_preprocessor(k), v)
                    for k, v in self.topic_word_weights[topic_idx]
                    .iloc[:n_words]
                    .to_dict()
                    .items()
                ]
            )
            # wc.recolor(color_func=kwargs['color_func'], random_state=random_state)
            plt.imshow(wc.recolor(color_func=color_func, random_state=random_state))
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

    def plot_topic_trajectory(self, urls):
        _topic_weights = self.topic_weights(urls)
        _topic_weights = _topic_weights.T / _topic_weights.max(axis=1)
        sns.heatmap(_topic_weights, cbar=False, linewidths=1)
        plt.ylabel('Topic')
        plt.xlabel('Page view')
        ax = plt.gca()
        start, end = ax.get_xlim()
        if _topic_weights.shape[1] > 20:
            ax.xaxis.set_ticks(arange(start, end, 10))
            ax.xaxis.set_ticklabels(arange(start, end, 10).astype(int))
        return ax

    def plot_topic_trajectory_of_tcid(self, tcid, data):
        d = data[data.tc_id == tcid].sort_values(by='timestamp', ascending=True)
        urls = d.data_url_test
        ax = self.plot_topic_trajectory(urls)
        conversion_idx = where(array(d.data_env_template == 'funnel_confirmation'))[0]
        if len(conversion_idx):
            min_y, max_y = plt.ylim()
            for idx in conversion_idx:
                plt.plot((idx + 0.5, idx + 0.5), (min_y, max_y), 'b-')
