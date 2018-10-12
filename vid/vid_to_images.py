from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import subprocess as sp
from warnings import warn
import datetime

FFMPEG_BIN = "ffmpeg"  # on Linux ans Mac OS
N_COLORS = 3
SECONDS_IN_A_DAY = 24 * 60 * 60
MAX_SECONDS = SECONDS_IN_A_DAY


def time_str_from_seconds(seconds):
    return datetime.datetime.utcfromtimestamp(seconds).strftime("%H:%M:%S.%f")


class VideoFile():
    def __init__(self, filepath,
                 video_dims=(1920, 1080),
                 start_s=0,
                 stop_s=None,
                 duration_s=None,
                 dtype='uint8'):

        self.filepath = filepath
        self.video_dims = video_dims
        self._video_frame_bytes = np.prod(video_dims) * N_COLORS
        self._video_rgb_mat_shape = tuple(list(video_dims) + [N_COLORS])

        self.start_s = start_s
        if duration_s is not None:
            if stop_s is not None:
                warn("You specified stop_s AND duration_s. IGNORING stop_s.")
            stop_s = start_s + duration_s
        elif stop_s is not None:
            duration_s = stop_s - start_s
        self.stop_s = stop_s
        self.duration_s = duration_s

        self.pipe = None

    def __enter__(self):
        command = [FFMPEG_BIN]

        if self.start_s is not None:
            if self.start_s < 0 or self.start_s >= MAX_SECONDS:
                raise ValueError(
                    "start_s must be positive and smaller than {}".format(MAX_SECONDS))
            command += ['-ss', time_str_from_seconds(self.start_s)]
        if self.stop_s is not None:
            if self.stop_s < 0 or self.stop_s >= MAX_SECONDS:
                raise ValueError(
                    "start_s must be positive and smaller than {}".format(MAX_SECONDS))
            command += ['-to', time_str_from_seconds(self.stop_s)]

        command += ['-i', self.filepath,
                    '-f', 'image2pipe',
                    '-pix_fmt', 'rgb24',
                    '-vcodec', 'rawvideo', '-']
        self.pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
        return self

    def __exit__(self, *args):
        self.pipe.kill()

    def read_frame_raw_image(self):
        return self.pipe.stdout.read(self._video_frame_bytes)

    def read_frame(self):
        # read video_dims bytes (= 1 frame)
        raw_image = self.read_frame_raw_image()
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape(self._video_rgb_mat_shape)
        # throw away the data in the pipe's buffer.
        self.pipe.stdout.flush()
        return image

    def skip_frames(self, n_frames=1):
        for i in xrange(n_frames):
            self.read_frame_raw_image()


top_mean_rgb = np.iinfo('uint8').max
color_of_idx = 'rgb'
idx_of_color = {c: i for i, c in enumerate(color_of_idx)}


def color_dominance_filt(im,
                         color='g',
                         min_mean_rgb=top_mean_rgb / float(N_COLORS),
                         max_mean_rgb=2 * top_mean_rgb / float(N_COLORS)):
    color_idx = idx_of_color[color]
    other_colors_idx = np.array([x for x in xrange(len(color_of_idx)) if x != color_idx])
    mean_rgb_mat = im.mean(axis=2)
    filtered_im = im[:, :, color_idx] / im[:, :, other_colors_idx].max(axis=2)

    filtered_im *= min_mean_rgb <= mean_rgb_mat
    filtered_im *= mean_rgb_mat <= max_mean_rgb
    return filtered_im


def get_image_and_filt_image(source,
                             offset_s=0,
                             filt_func=color_dominance_filt,
                             filt_kwargs=None,
                             figsize=(8, 4)
                             ):
    if filt_kwargs is None:
        filt_kwargs = dict(color='g',
                           min_mean_rgb=70,
                           max_mean_rgb=200)
    with VideoFile(source, start_s=offset_s) as fp:
        im = fp.read_frame()

    filtered_im = filt_func(im, **filt_kwargs)

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(121).imshow(im)
    plt.xticks([])
    plt.yticks([])
    fig.add_subplot(122).imshow(filtered_im, cmap='gray_r')
    plt.xticks([])
    plt.yticks([])

    return im, filtered_im
