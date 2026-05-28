"""
Test region detection using GADM shapefile for Cameroon.
"""

from re import match

import geopandas as gpd
from shapely.geometry import Point
import os

# ─── Path to your shapefile ───────────────────────────────────────────────────
SHAPEFILE_PATH = r"C:\Users\mike\Downloads\camagri-platform\data\gadm41_CMR_shp\gadm41_CMR_1.shp"

# ─── Test coordinates ─────────────────────────────────────────────────────────
TEST_POINTS = {
    "Yaoundé (Centre)":    (3.848,  11.502),
    "Douala (Littoral)":   (4.050,   9.700),
    "Bamenda (North West)":(5.959,  10.146),
    "Garoua (North)":      (9.300,  13.397),
    "Bafoussam (West)":    (5.478,  10.417),
    "Buea (South West)":   (4.154,   9.243),
    "Bertoua (East)":      (4.578,  13.685),
    "Ebolowa (South)":     (2.900,  11.150),
    "Ngaoundéré (Adamawa)":(7.321,  13.584),
    "Maroua (Far North)":  (10.591, 14.315),
}

def test_region_detection(lat,lon, path=SHAPEFILE_PATH):
    if not os.path.exists(path):
        return "Center"
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    detected = Point(lon, lat)  # shapely is (lon, lat)
    match = gdf[gdf.geometry.contains(detected)]
    if not match.empty:
        return match.iloc[0]["NAME_1"]
    return detected, "Center"

if __name__ == "__main__":
    print(test_region_detection(4.146735173124221,9.292270559690008))