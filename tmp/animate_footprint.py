from skyfield.api import load, wgs84
from datetime import timedelta
import math
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely import Polygon, coverage_union, normalize


mpl.font_manager.fontManager.addfont('THSarabunChula-Regular.ttf')
font = {'family' : 'TH Sarabun Chula',
        'weight' : 'bold',
        'size'   : 16}
mpl.rc('font', **font)

ts = load.timescale()


MINIMUM_ELEVATION_ANGLE = 10.0 # should be dynamic (will update in lv.4)
EARTH_RADIUS = 6378 # in km


def load_satellites(filepath):
    satellites = load.tle_file(filepath)
    return {sat.name: sat for sat in satellites}


def animate_util(point_of_time_i, timeseries_footprints, polygons):
    for polygon, timeseries_footprint in zip(polygons, timeseries_footprints):
        footprint = np.array(timeseries_footprint[point_of_time_i]['footprint'])
        polygon.set_data(footprint[:, 1], footprint[:, 0])


def animate_globalmap(timeseries_footprints):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    ax.stock_img()
    ax.coastlines()

    polygons = [
        ax.plot([], [], color='blue', transform=ccrs.PlateCarree())[0] 
        for _ in timeseries_footprints
    ]

    ani = animation.FuncAnimation(
        fig, animate_util, len(timeseries_footprints[0]), 
        fargs=(
            timeseries_footprints,
            polygons,
        ),
        interval=100
    )

    plt.show()


def euclidean_distance(position1, position2):
    if len(position1) != len(position2):
        raise Exception("len(position1) != len(position2)")
    
    distance = sum([(u-v)**2 for u, v in zip(position1, position2)]) ** 0.5
    return distance


def haversine(ref_lat, ref_lon, theta):
    ref_lat = ref_lat.radians
    ref_lon = ref_lon.radians
    theta = math.radians(theta)

    polygon = []

    for bearing in range(360):
        bearing = math.radians(bearing)
        new_lat = math.asin(math.sin(ref_lat) * math.cos(theta) + math.cos(ref_lat) * math.sin(theta) * math.cos(bearing))
        new_lon = ref_lon + math.atan2(math.sin(bearing) * math.sin(theta) * math.cos(ref_lat), math.cos(theta) - math.sin(ref_lat) * math.sin(new_lat))
        polygon.append((math.degrees(new_lat), math.degrees(new_lon)))

    return polygon


def get_footprint_at(t, satellite):
    geocentric = satellite.at(t)
    lat, lon = wgs84.latlon_of(geocentric)

    distance_from_origin = euclidean_distance(geocentric.position.km, [0, 0, 0])
    theta = math.degrees(math.acos(
        EARTH_RADIUS / distance_from_origin * math.cos(math.radians(MINIMUM_ELEVATION_ANGLE))
        ) - math.radians(MINIMUM_ELEVATION_ANGLE))
    
    footprint = haversine(lat, lon, theta)
    
    return {
        'name': satellite.name,
        'datetime': t,
        'geocentric': geocentric.position.km,
        'latitude': lat,
        'longitude': lon,
        'theta': theta,
        'footprint': footprint
    }


def is_intersect(polygon1, polygon2):
    polygon1 = Polygon(polygon1)
    polygon2 = Polygon(polygon2)

    print(polygon1.intersects(polygon2))


def main():
    t0 = ts.now() # get current time
    duration = timedelta(days=1)
    t1 = t0 + duration

    satellites = load_satellites('TLE/THEOS1.txt')
    print(f'Loaded {len(satellites)} satellites')

    print('Calculate Satellite Footprint...')
    timeseries_footprints = []
    
    for satellite in satellites.values():
        timeseries_footprints.append([
            get_footprint_at(t0 + timedelta(minutes=i), satellite)
            for i in range(int(duration.total_seconds()/60) + 1)
        ])
        print(satellite.name, '(finished)')

    animate_globalmap(timeseries_footprints)
    
    
main()