__author__ = 'thorwhalen'

import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import os
import fnmatch
import numpy as np
from matplotlib import animation


def mk_2d_sequence_gif(x_seq, y_seq, filename='make_2d_sequence_gif.gif', plot_kwargs={},
                       edit_funs={}, writeGif_kwargs={}, savefig_kwargs={}, **kwargs):
    from images2gif import writeGif

    # handling defaults
    plot_kwargs = dict({'color': 'b', 'marker': 'o', 'linestyle': '-', 'linewidth': 0.2}, **plot_kwargs)
    savefig_kwargs = dict({'dpi': 200}, **savefig_kwargs)

    writeGif_kwargs = dict({'size': (600, 600), 'duration': 0.2}, **writeGif_kwargs)

    # handling edit_funs
    if hasattr(edit_funs, '__call__'):
        edit_fun = edit_funs
    else:
        # defining an edit function from the edit_funs dict
        def edit_fun():  # this function assumes keys of edit_funs are plt attributes, and tries to call the values on them
            for k, v in edit_funs.items():
                if hasattr(plt, edit_funs[k]):
                    f = plt.__getattribute__(k)
                    try:
                        f(**v)
                    except:
                        try:
                            f(*v)
                        except:
                            f(v)
                else:
                    v()

    tmp_dir = tempfile.mkdtemp('mk_2d_sequence_gif')

    # get the axis limits
    if 'xylims' in list(kwargs.keys()):
        ax = kwargs['xylims']
    else:
        # plot the full sequences to get the axis limits from it
        plt.plot(x_seq, y_seq, **plot_kwargs)
        edit_fun()
        ax = plt.axis()
    # create files that will be absorbed by the gif
    for i in range(1, len(x_seq) + 1):
        plt.delaxes()
        plt.plot(x_seq[:i], y_seq[:i], **plot_kwargs)
        plt.axis(ax)
        edit_fun()
        plt.savefig(os.path.join(tmp_dir, 'mk_2d_sequence_gif%02.0f' % i), **savefig_kwargs)
    # create the gif
    size = writeGif_kwargs['size']
    del writeGif_kwargs['size']
    file_names = sorted(fnmatch.filter(os.listdir(tmp_dir), 'mk_2d_sequence_gif*'))
    images = [Image.open(os.path.join(tmp_dir, fn)) for fn in file_names]
    for im in images:
        im.thumbnail(size, Image.ANTIALIAS)
    writeGif(filename=filename, images=images, **writeGif_kwargs)


def mk_rank_progression_bars(scores, save_filepath='rank_progression_bars.mp4',
                             figsize=(16, 5), markersize=30, ms_between_frames=100, fps=10):
    """

    :param scores:
    :param save_filepath:
    :param figsize:
    :param markersize:
    :param ms_between_frames:
    :param fps:
    :return:
    >>> scores = np.random.permutation(20)
    >>> mk_rank_progression_bars(scores)
    """

    def barlist(i, scores):
        return [len(scores) - score if score <= i else 0 for score in scores]

    fig = plt.figure(figsize=figsize)

    n = len(scores)
    plt.xlim((-0.5, n))
    plt.ylim((0, n))

    def animate(i):
        y = np.array(barlist(i, scores))
        ax = plt.gca()
        ax.clear()
        plt.bar(range(len(y)), y, color='k')
        if markersize > 0:
            n = len(scores)
            new_score = np.min(y[y > 0])
            new_score_idx = np.where(y == new_score)[0][0]
            plt.plot(new_score_idx, n * (1.05), 'vb', markersize=markersize)

    anim = animation.FuncAnimation(
        fig, animate, repeat=False, blit=False, frames=len(scores), interval=ms_between_frames)

    anim.save(save_filepath, writer=animation.FFMpegWriter(fps=fps));
