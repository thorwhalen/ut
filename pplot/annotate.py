__author__ = 'thor'



def xlabel_inside(ax, text):
    ax.text(.5, 0.01, text,
        horizontalalignment='center',
        verticalalignment='bottom',
        rotation=None,
        transform=ax.transAxes)


def ylabel_inside(ax, text):
    ax.text(.01, 0.5, text,
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)