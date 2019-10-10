__author__ = 'thor'
"""
Accessing information on population in the word.

In order to work, you need to have a mongo collection (by default named geo_pop_density)
in a db (by default called util), with the appropriate datas there.

The way I processed the data:

Downloaded  and unzipped data from:
    http://sedac.ciesin.columbia.edu/data/set/gpw-v3-centroids/data-download

To avoid utf8 errors in import, I did iconv -f ISO-8859-1 -t utf-8 gl_centroids.csv > gl_centroids_utf8.csv

Then call _import_data_into_mongo on the filename (see options in function)

See:
    Indexing:
        http://docs.mongodb.org/manual/tutorial/build-a-2d-index/#geospatial-indexes-range
        http://docs.mongodb.org/manual/core/geospatial-indexes/#geospatial-indexes-geohash

"""
from pymongo import MongoClient
from pymongo.errors import InvalidStringData
from geopy.distance import vincenty
from numpy import *
import numpy as np
import pandas as pd
import re
from math import ceil, log, fmod


from ut.iacc.geo.gutil import GeoMongoDacc
from ut.pstr.trans import str_to_utf8_or_bust
from ut.util.log import printProgress


class Geopop(object):
    def __init__(self, pop_db='util', pop_col='geo_pop_density',
                 coordinate_field='cen', pop_field='p15a', pop_dens_field='p15adens', area_field='areasqkm'):
        self.mdacc = GeoMongoDacc(db=pop_db, collection=pop_col, coordinate_field=coordinate_field)
        self.pop_field = pop_field
        self.coordinate_field = coordinate_field
        self.pop_dens_field = pop_dens_field
        self.area_field = area_field

    def population_of_nearest_latlon(self, lat, lon):
        return self.mdacc.find_nearest_one(lat=lat, lon=lon, fields=[self.pop_field])[self.pop_field]

    def within_radius_cursor(self, lat, lon, radius_km, fields=None):
        if fields is None:
            fields = {'_id': False,
                      self.pop_field: True,
                      self.coordinate_field: True,
                      self.pop_dens_field: True,
                      self.area_field: True}
        return self.mdacc.find_in_circle(
                lat=lat,
                lon=lon,
                max_km=radius_km,
                fields=fields)

    def population_within_radius(self, lat, lon, radius_km):
        """
        Estimates the population within a circle whose center is (lat,lon) and radius is radius_km.
        It does so by
        """
        cursor = self.within_radius_cursor(lat, lon, radius_km, fields=[self.pop_field])
        pop = self.total_population(cursor)
        if pop != 0:
            return pop
        else:
            d = self.mdacc.find_nearest_one(lat=lat, lon=lon, fields=[self.pop_field, self.area_field])
            total_population = d[self.pop_field]
            total_area = d[self.area_field]
            circle_area = np.pi * radius_km ** 2
            return circle_area * total_population / total_area

    def population_within_radius_density_approach_not_verified(self, lat, lon, radius_km):
        """
        Estimates the population within a circle whose center is (lat,lon) and radius is radius_km.
        It does so by estimating the population density in the area, as the total of the centroid populations divided by
          the total of the surface areas covered by the centroids, and multiplying this by the surface of the queried
          circle.

        --> I haven't verified if this would work well near water surfaces.
        --> Perhaps it would be better (though less efficient) to use gridded stats instead of centroid stats.
        """
        cursor = self.within_radius_cursor(lat, lon, radius_km,
                                           fields={self.pop_field: True,
                                                   self.coordinate_field: True,
                                                   self.area_field: True})
        d = pd.DataFrame([x for x in cursor])

        if len(d) == 0:
            d = self.mdacc.find_nearest_one(lat=lat, lon=lon, fields=[self.pop_field, self.area_field])
            total_population = d[self.pop_field]
            total_area = d[self.area_field]
        else:
            total_population = sum(d[self.pop_field])
            total_area = sum(d[self.area_field])

        circle_area = np.pi * (radius_km ** 2)
        # print len(d), d
        # print circle_area, total_population, total_area
        return circle_area * total_population / total_area

    def population_within_radius_simple(self, lat, lon, radius_km):
        cursor = self.within_radius_cursor(lat, lon, radius_km, fields=[self.pop_field])
        return self.total_population(cursor)

    def population_within_radius_approximation(self, lat, lon, radius_km):
        d = self.mdacc.find_nearest_one(lat=lat, lon=lon, fields=[self.pop_field, self.coordinate_field])
        return d

    def fuzzy_population_within_radius(self, lat, lon, radius_km):
        """

        """
        cursor = self.within_radius_cursor(lat, lon, radius_km)
        d = pd.DataFrame([x for x in cursor])
        center = (lat, lon)
        if len(d) != 0:
            distance = list(map(lambda xlat, xlon: vincenty((xlat, xlon), center).kilometers,
                                                    [x[1] for x in d[self.coordinate_field]],
                                                    [x[0] for x in d[self.coordinate_field]]))
            membership = [max(0, (radius_km - x) / radius_km) for x in distance]
            return sum(membership * d[self.pop_field])
        else:
            d = self.mdacc.find_nearest_one(lat=lat, lon=lon, fields=[self.pop_field, self.area_field])
            total_population = d[self.pop_field]
            total_area = d[self.area_field]
            circle_area = np.pi * (radius_km ** 2)
            cone_area_population = circle_area * total_population / total_area
            return cone_area_population / 3  # volume of cone of height 1 is 1/3 of area of circle base

    #
    # def fuzzy_population(self, geo_pop_dict, radius):
    #     km_distance = vincenty(x[self.coordinate_field][1], x[self.coordinate_field][0]).kilometers

    def total_population(self, item_iterator):
        return float(sum([x[self.pop_field] for x in item_iterator]))

_meters_to_bits = lambda meters: int(min(32, ceil(log(0.6 / meters, 2) + 26)))


def _import_data_into_mongo(filepath='gl_centroids_utf8.csv',
                            mongo_db='util',
                            mongo_collection='geo_pop_density',
                            index_precision_meters=76.8,
                            print_mongo_import_progress_every=50000):

    bits = _meters_to_bits(index_precision_meters)
    printProgress("importing %s into dataframe" % filepath)
    d = pd.read_csv(filepath, header=0, sep=',', quotechar="'")
    space_re = re.compile('\s')
    d.columns = [space_re.sub('_', str(x).lower()) for x in d.columns]  # I want lower and no-space_columns

    printProgress("importing dataframe rows into mongo (will print progress every %d items"
                  % print_mongo_import_progress_every)
    mc = MongoClient()
    db = mc[mongo_db]
    db.drop_collection(mongo_collection)
    db.create_collection(mongo_collection)
    mg_collection = db[mongo_collection]

    n = len(d)
    for i, di in enumerate(d.iterrows()):
        ddi = _process_dict(dict(di[1]))
        if fmod(i, print_mongo_import_progress_every) == 0:
            printProgress("  %d/%d" % (i, n))
        try:
            mg_collection.insert(ddi, w=0)
        except InvalidStringData:
            ddi = {k: str_to_utf8_or_bust(v) for k, v in ddi.items()}
            mg_collection.insert(ddi, w=0)

    printProgress("ensuring GEOSPHERE index with %d bits (for a precision of %d meters or more"
                  % (bits, index_precision_meters))
    from pymongo import GEOSPHERE
    mg_collection.ensure_index([("cen", GEOSPHERE), ("bits", bits)])
    printProgress("------------------------------ DONE ------------------------------")


def _process_dict(d):
    # replace the _id (it's the only exception to the rest of this,
    #    and will be recreated when dumped anyway)
    d.pop('_id', None) # in case d comes from a mongo dict
    d.pop('adminid', None) # it's frequently bull shit, so drop it
    d['country'] = d['countrynm']  # I prefer to use country then countrynm
    d.pop('countrynm')
    # remove first and last single quotes from string values
    d.update({k: str_to_utf8_or_bust(d[k]) for k in list(d.keys()) if isinstance(d[k], str)})
    # do the same for keys, and while we're at it, remove "N.A" keys (and while we're at it, lower case keys)
    d = {k.lower(): v for k, v in d.items() if v != "N.A."}
    # make location keys
    try:
        if not isinstance(d["lat_cen"], float) \
                or not isinstance(d["long_cen"], float) \
                or not isinstance(d["lat_lbl"], float) \
                or not isinstance(d["long_lbl"], float):
            return None
        else:
            d["cen"] = [d["long_cen"], d["lat_cen"]]
            d["lbl"] = [d["long_lbl"], d["lat_lbl"]]
    except KeyError:
        return None
    # remove the original location fields
    d.pop("lat_cen")
    d.pop("long_cen")
    d.pop("lat_lbl")
    d.pop("long_lbl")
    return d

