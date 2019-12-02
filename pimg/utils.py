import numpy as np
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import matplotlib.pylab as plt
from io import BytesIO
import os


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


def func_to_get_pil_images_like_this_one(x, **kwargs):
    """Get a function that will convert images, specified as the example input x, into PIL.Image types"""
    if isinstance(x, Image.Image):
        def to_pil_image(x):
            return x
    elif isinstance(x, plt.Figure):
        def to_pil_image(x):
            fp = BytesIO()
            kwargs['format'] = kwargs.get('format', 'jpg')
            x.savefig(fp, **kwargs)
            return Image.open(fp)
    else:
        to_pil_image = Image.open

    return to_pil_image


def x_to_pil_image(x, **kwargs):
    to_pil_image = func_to_get_pil_images_like_this_one(x, **kwargs)
    return to_pil_image(x)


def write_images(images, fp='test.pdf', pil_write_format=None, to_pil_image_kwargs=None, **pil_save_kwargs):
    """Write images or figs into pdf pages (one image per page)

    Args:
        images: iterable of "images" (should be a matplotlib figure, a PIL.Image,
            or a type that PIL.Image.open can open")
        fp: The filepath, path.Path, or file object to dump the images into
        **params: # extra parameters that are passed on to the save function

    Returns:

    """
    to_pil_image_kwargs = to_pil_image_kwargs or {}
    _, ext_format = os.path.splitext(fp)
    if ext_format.startswith('.'):
        ext_format = ext_format[1:]
    pil_write_format = pil_write_format or ext_format or 'PDF'
    im_iter = iter(images)
    first_im = next(im_iter)
    to_pil_image = func_to_get_pil_images_like_this_one(first_im, **to_pil_image_kwargs)
    return to_pil_image(first_im).save(fp, format=pil_write_format, save_all=True,
                                       append_images=map(to_pil_image, im_iter), **pil_save_kwargs)
