

import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from ut.pimg.utils import crop_out_identical_margins

folder_containing_current_file, _ = os.path.split(__file__)
ttf_font_folder = os.path.join(folder_containing_current_file, 'data/fonts/ttf/')

# DEFAULT_FONT_FILE = 'DejaVuSans-Bold.ttf'
DEFAULT_FONT_FILE = 'courier.ttf'

def local_font_files():
    return [f for f in os.listdir(ttf_font_folder) if f.endswith('.ttf')]


def text2img(text,
             color="black",
             bgcolor=None,
             font_file=DEFAULT_FONT_FILE,
             vspace=20,
             fontsize=50,
             width=600,
             x_margin=0,
             y_margin=0):
    """

    >>> img = text2img('asdf ' * 40, color='white', fontsize=40, x_margin=0, y_margin=0, width=600)
    >>> img.save(open('asdf.png', 'w'))
    :param text:
    :param color:
    :param bgcolor:
    :param fontfullpath:
    :param vspace:
    :param fontsize:
    :param width:
    :param x_margin:
    :param y_margin:
    :return:
    """

    if color in ['w', 'white']:
        color = "#FFF"
    elif color in ['k', 'black']:
        color = "#000"
    if bgcolor is None:
        if color == "#000":
            bgcolor = '#FFF'
        else:
            bgcolor = "#000"

    if font_file is not None and not os.path.isfile(font_file):
        font_file = os.path.join(ttf_font_folder, font_file)

    REPLACEMENT_CHARACTER = '\uFFFD'
    NEWLINE_REPLACEMENT_STRING = ' ' + REPLACEMENT_CHARACTER + ' '

    font = ImageFont.load_default() if font_file == None \
        else ImageFont.truetype(font_file, fontsize)
    text = text.replace('\n', NEWLINE_REPLACEMENT_STRING)

    lines = []
    line = ""

    for word in text.split():
        if word == REPLACEMENT_CHARACTER:  # give a blank line
            lines.append(line[1:])  # slice the white space in the begining of the line
            line = ""
            lines.append("")  # the blank line
        elif font.getsize(line + ' ' + word)[0] <= (width):
            line += ' ' + word
        else:  # start a new line
            lines.append(line[1:])  # slice the white space in the begining of the line
            line = ""

            # TODO: handle too long words at this point
            line += ' ' + word  # for now, assume no word alone can exceed the line width

    if len(line) != 0:
        lines.append(line[1:])  # add the last line

    line_height = font.getsize(text)[1]
    img_height = line_height * (len(lines) + 1)
    img_width = max([font.getsize(line)[0] for line in lines])
    print(img_width)

    img = Image.new("RGBA", (img_width, img_height), bgcolor)
    draw = ImageDraw.Draw(img)

    y = 0
    for line in lines:
        draw.text((0, y), line, color, font=font)
        y += vspace + line_height / 2

    img = crop_out_identical_margins(img, x_margin=x_margin, y_margin=y_margin)

    return img
