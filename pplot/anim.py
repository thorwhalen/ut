__author__ = 'thorwhalen'

import matplotlib.pyplot as plt
import tempfile
from images2gif import writeGif
from PIL import Image
import os
import fnmatch

def mk_2d_sequence_gif(x_seq, y_seq, filename='make_2d_sequence_gif.gif', plot_kwargs={}, edit_funs={}, writeGif_kwargs={}, savefig_kwargs={}, **kwargs):
    # handling defaults
    plot_kwargs = dict({'color':'b', 'marker':'o', 'linestyle':'-', 'linewidth':0.2}, **plot_kwargs)
    savefig_kwargs = dict({'dpi':200}, **savefig_kwargs)

    writeGif_kwargs = dict({'size':(600,600), 'duration':0.2}, **writeGif_kwargs)

    # handling edit_funs
    if hasattr(edit_funs, '__call__'):
        edit_fun = edit_funs
    else:
    # defining an edit function from the edit_funs dict
        def edit_fun(): # this function assumes keys of edit_funs are plt attributes, and tries to call the values on them
            for k, v in edit_funs.iteritems():
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
    if 'xylims' in kwargs.keys():
        ax = kwargs['xylims']
    else:
        # plot the full sequences to get the axis limits from it
        plt.plot(x_seq, y_seq, **plot_kwargs)
        edit_fun()
        ax = plt.axis()
    # create files that will be absorbed by the gif
    for i in range(1,len(x_seq)+1):
        plt.delaxes()
        plt.plot(x_seq[:i], y_seq[:i], **plot_kwargs)
        plt.axis(ax)
        edit_fun()
        plt.savefig(os.path.join(tmp_dir, 'mk_2d_sequence_gif%02.0f'%i), **savefig_kwargs)
    # create the gif
    size = writeGif_kwargs['size']
    del writeGif_kwargs['size']
    file_names = sorted(fnmatch.filter(os.listdir(tmp_dir), 'mk_2d_sequence_gif*'))
    images = [Image.open(os.path.join(tmp_dir,fn)) for fn in file_names]
    for im in images:
        im.thumbnail(size, Image.ANTIALIAS)
    writeGif(filename=filename, images=images, **writeGif_kwargs)


