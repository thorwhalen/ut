__author__ = 'thor'

import matplotlib.pyplot as plt
import numpy as np
import scipy


def xy_density(xdat, ydat, cmap='jet', marker='.', imshow_kwargs={},
               bins=[100, 100], density_thresh=0, xyrange=None, plot_kwargs={}):
    '''
    graphs the density of (x,y) points in the plane, using color (defined by cmap) except when density is below
    density_thresh, in which case the points themselves are ploted
    '''

    # validation
    assert len(xdat) == len(ydat), 'xdat and ydat must have the same length'

    #histogram definition
    xyrange = xyrange or [[np.min(xdat), np.max(xdat)], [np.min(ydat), np.max(ydat)]] # data range
    if isinstance(bins, int):
        bins = [bins, bins]
    # else:
    #     if isinstance(bins[0], int):
    #         bins[0] = range(bins[0])
    #     if isinstance(bins[1], int):
    #         bins[1] = range(bins[1])
    if density_thresh < 1:  # if density_thresh is a float, interpret it as a ratio of the number of bins
        density_thresh = density_thresh * bins[0] * bins[1]

    # histogram the data
    hh, locx, locy = scipy.histogram2d(xdat, ydat, range=xyrange, bins=bins)
    # return hh, locx, locy
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    #select points within the histogram
    # ind = (posx > 0) & (posx <= np.max(bins[0])) & (posy > 0) & (posy <= np.max(bins[1]))
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < density_thresh] # low density points
    ydat1 = ydat[ind][hhsub < density_thresh]
    # hh[hh < density_thresh] = np.nan # fill the areas with low density by NaNs

    plt.imshow(np.flipud(hh.T), cmap=cmap, interpolation='none', extent=np.array(xyrange).flatten(), **imshow_kwargs)
    plt.colorbar()
    plot_kwargs = dict({'color': plt.cm.get_cmap(cmap)(0)}, **plot_kwargs)
    plt.plot(xdat1, ydat1, marker, **plot_kwargs)
    plt.show()
    return hh