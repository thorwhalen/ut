__author__ = 'thor'


from pymongo import MongoClient
from bson import SON


degree_kms = 111.12


class GeoMongoDacc(object):
    def __init__(self, db, collection, coordinate_field):
        self.collection = MongoClient()[db][collection]
        self.coordinate_field = coordinate_field

    def find_nearest_one(self, lat, lon, **kwargs):
        return self.collection.find_one(
            self.nearest_neighbors_query(coordinate_field=self.coordinate_field, lat=lat, lon=lon),
            **kwargs
        )

    def find_nearest(self, lat, lon, **kwargs):
        return self.collection.find(
            self.nearest_neighbors_query(coordinate_field=self.coordinate_field, lat=lat, lon=lon),
            **kwargs
        )

    def find_in_circle(self, lat, lon, max_km, **kwargs):
        return self.collection.find(
            self.in_circle_query(coordinate_field=self.coordinate_field, lat=lat, lon=lon, max_km=max_km),
            **kwargs
        )


    @classmethod
    def nearest_neighbors_query(cls, coordinate_field, lat, lon):
        return {coordinate_field: SON([("$near", {"$geometry": SON([("type", "Point"),
                                                                    ("coordinates", [lon, lat])])})])}


    @classmethod
    def in_circle_query(cls, coordinate_field, lat, lon, max_km):
        return {coordinate_field: {"$within": {"$center": [[lon, lat], max_km / degree_kms]}}}