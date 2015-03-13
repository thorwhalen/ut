__author__ = 'thor'


from numpy import *
from mpl_toolkits.basemap import Basemap


def map_records(d, basemap_kwargs={}, plot_kwargs={}, lat_col='latitude', lng_col='longitude'):
    # Create the Basemap
    basemap_kwargs = dict(dict(projection='merc', # there are other choices though
                               resolution='l',  # c(rude), l(ow), i(ntermediate), h(igh), and f(ull)
                               area_thresh=1000.0,
                               llcrnrlat=max([-89.999999, min(d[lat_col])]),
                               llcrnrlon=min(d[lng_col]),  # Lower left corner
                               urcrnrlat=min([89.999999, max(d[lat_col])]),
                               urcrnrlon=max(d[lng_col]) # Upper right corner
                               ),
                          **basemap_kwargs)
    event_map = Basemap(**basemap_kwargs)

    # Draw important features
    event_map.drawcoastlines()
    event_map.drawcountries()
    event_map.fillcontinents(color='0.8') # Light gray
    event_map.drawmapboundary()

    plot_kwargs = dict(dict(marker='o',
                            markersize=7,
                            color='b',
                            alpha=0.1),
                        **plot_kwargs)
    y, x = event_map(array(d[lng_col]), array(d[lat_col]))
    event_map.plot(y, x, 'bo', **plot_kwargs)

    return event_map

