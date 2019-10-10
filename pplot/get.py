__author__ = 'thor'



def get_colorbar_tick_labels_as_floats(cbar):
    return [float(ww.get_text().replace('\u2212', '-')) for ww in cbar.ax.yaxis.get_majorticklabels()]


