from __future__ import division

import numpy as np
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def crop_out_identical_margins(img, x_margin=0, y_margin=0):
    """
    Returns a cropped image of the input, where the "uninformative" margins of the images are cropped out.

    :param img: A PIL.Image.Image
    :return: A PIL.Image.Image where the margins that are identical (i.e. same exact "color+alpha") are cropped out
    """
    ima = np.asarray(img)
    if len(ima.shape) > 2:
        ima = ima.sum(axis=2)

    x_size = ima.shape[1]
    y_size = ima.shape[0]

    min_x = np.where(np.diff(ima.sum(axis=0)) != 0)[0][0]
    min_x -= min(min_x, x_margin)
    min_y = np.where(np.diff(ima.sum(axis=1)) != 0)[0][0]
    min_y -= min(min_y, y_margin)

    max_x = np.where(np.diff(ima.sum(axis=0)) != 0)[0][-1] + 1
    max_x += min(x_size - max_x, x_margin)
    max_y = np.where(np.diff(ima.sum(axis=1)) != 0)[0][-1] + 1
    max_y += min(y_size - max_y, y_margin)

    print((x_size, y_size))
    print((min_x, min_y, max_x, max_y))
    return img.crop((min_x, min_y, max_x, max_y))
