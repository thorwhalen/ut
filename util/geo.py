__author__ = 'thorwhalen'

from sklearn.neighbors import NearestNeighbors
import numpy as np
import math

earth_radius_mi = 3959.0
earth_radius_km = 6371.0


########################################################################################################################
# EARTH GEOMETRY

def find_closest_geo_locations(latlons, geo_location_latlons, n_neighbors=1, earth_radius=None, max_radius=None):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='haversine').fit(np.deg2rad(geo_location_latlons))
    distances, indices = nbrs.kneighbors(np.deg2rad(latlons))
    if earth_radius:
        distances = distances * earth_radius
    if max_radius:
        distances, indices = list(zip(*list(map(_radius_filter, distances, indices, [max_radius]*len(distances)))))
    return distances, indices


def _radius_filter(distances, indices, max_radius):
    lidx = np.array(distances) < max_radius
    return np.array(distances)[lidx], np.array(indices)[lidx]


def keep_only_distances_and_indices_within_radius(max_radius, distances, indices):
    lidx = distances < max_radius
    distances = list(map(lambda x, y: x[y], distances, lidx))
    indices = list(map(lambda x, y: x[y], indices, lidx))
    return (distances, indices)


def rad2km(distances):
    return distances * earth_radius_km


def rad2mi(distances):
    return distances * earth_radius_mi


def mi2km(distances):
    return distances * earth_radius_km / earth_radius_mi


def km2mi(distances):
    return distances * earth_radius_mi / earth_radius_km


def rad_distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos_value = math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) + math.cos(phi1) * math.cos(phi2)
    if cos_value > 1:
        cos_value = 1.0
    arc = math.acos(cos_value)

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc


