from skyfield.api import load, wgs84
from datetime import timedelta
import math
import numpy as np
import seaborn as sns
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely import Polygon, MultiPolygon


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


def globalmap(ax, timeseries_footprints):
    ax.stock_img()
    ax.coastlines()
    ax.title.set_text('Animate Satellite Footprint of Satellite Constellation')

    polygons = [
        ax.plot([], [], color='blue', transform=ccrs.PlateCarree())[0] 
        for _ in timeseries_footprints
    ]

    return polygons


def get_percent_coverages(timeseries_footprints, preferred_region):
    n_point_of_time = len(timeseries_footprints[0])
    n_satellite = len(timeseries_footprints)

    preferred_polygon = Polygon(preferred_region)
    percent_coverages = []

    for point_of_time_i in range(n_point_of_time):
        multi_polygon = []

        for satellite_i in range(n_satellite):
            footprint = timeseries_footprints[satellite_i][point_of_time_i]['footprint']
            polygon = Polygon([(lon, lat) for lat, lon in footprint])
            multi_polygon.append(polygon)
    
        union_polygon = Polygon()
        for polygon in multi_polygon:
            union_polygon = union_polygon | polygon
        difference = preferred_polygon - union_polygon

        percent_coverage = (preferred_polygon.area-difference.area) / preferred_polygon.area * 100
        percent_coverages.append(percent_coverage)

    return percent_coverages


def heatmap(ax, percent_coverages, start_time):
    sns.heatmap([percent_coverages], annot=False, cmap='Greens', ax=ax)
    ax.yaxis.set_ticks([]) # hide y-axis tick

    ax.title.set_text('ความครอบคลุมพื้นที่ประเทศไทยของกลุ่มดาวเทียมในหนึ่งวัน (%)')
    ax.set_xlabel(f"Minutes from {start_time.utc_strftime('%Y %b %d %H:%M:%S (UTC)')}")


def plot(timeseries_footprints, preferred_region):
    fig1 = plt.figure(1, figsize=(10, 5))
    fig2 = plt.figure(2, figsize=(7, 6))
    fig1_ax = fig1.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    fig2_ax = fig2.add_subplot(1, 1, 1)

    polygons = globalmap(fig1_ax, timeseries_footprints)
    percent_coverages = get_percent_coverages(timeseries_footprints, preferred_region)
    start_time = timeseries_footprints[0][0]['datetime']
    heatmap(fig2_ax, percent_coverages, start_time)

    ani = animation.FuncAnimation(
        fig1, animate_util, len(timeseries_footprints[0]), 
        fargs=(
            timeseries_footprints,
            polygons,
        ),
        interval=100
    )

    plt.show()


def main():
    t0 = ts.now() # get current time
    duration = timedelta(days=1)
    t1 = t0 + duration

    with open('polygon/thailand_polygon.txt', 'r') as f:
        thailand = [line.split(', ') for line in f.readlines() if line!='']

    satellites = load_satellites('TLE/test.txt')
    print(f'Loaded {len(satellites)} satellites')

    print('Calculate Satellite Footprint...')
    timeseries_footprints = []
    
    for satellite in satellites.values():
        timeseries_footprints.append([
            get_footprint_at(t0 + timedelta(minutes=i), satellite)
            for i in range(int(duration.total_seconds()/60) + 1)
        ])
        print(satellite.name, '(finished)')

    plot(timeseries_footprints, thailand)
    
    
main()