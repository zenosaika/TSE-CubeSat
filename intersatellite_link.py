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


def animate_util(point_of_time_i, timeseries_footprints, polygons, timeseries_ISL_graph, graph, fig1_ax):
    for polygon, timeseries_footprint in zip(polygons, timeseries_footprints):
        footprint = np.array(timeseries_footprint[point_of_time_i]['footprint'])
        polygon.set_data(footprint[:, 1], footprint[:, 0])

    # reset graph
    for edge in graph:
        edge.remove()
    graph.clear()

    # add edges to graph
    for src, dsts in timeseries_ISL_graph[point_of_time_i].items():
        for dst in dsts:
            src_lat = timeseries_footprints[src][point_of_time_i]['latitude'].degrees
            src_lon = timeseries_footprints[src][point_of_time_i]['longitude'].degrees
            dst_lat = timeseries_footprints[dst][point_of_time_i]['latitude'].degrees
            dst_lon = timeseries_footprints[dst][point_of_time_i]['longitude'].degrees
            graph.append(fig1_ax.plot([src_lon, dst_lon], [src_lat, dst_lat], color='y', marker='o', mfc='k', mec='w', transform=ccrs.PlateCarree())[0])


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


def get_possible_ISL_distance():
    # Radio Link Budget Calculation
    f = 433 * 10**6 # Hz
    EIRP = 3 # dB
    Gr = 3 # dBi
    EbN0 = 14 # dB
    R = 50 # dBbit/s
    N0 = -204 # dB
    M = 0 # dB

    Ls = EIRP + Gr - EbN0 - R - N0 - M
    possible_distance = 10**((Ls - 20*math.log10(f) + 147.55) / 20)

    return possible_distance / 1000 # in kilos


# undirected & weighted graph
def make_timeseries_ISL_graph(timeseries_footprints):
    possible_distance = get_possible_ISL_distance() # in meters
    duration_in_minutes = len(timeseries_footprints[0])
    number_of_satellites = len(timeseries_footprints)
    
    timeseries_ISL_graph = [] # list of adjacency list

    for t in range(duration_in_minutes):
        graph = {}
        for i in range(0, number_of_satellites):
            for j in range(i+1, number_of_satellites):
                ith_satellite_position = timeseries_footprints[i][t]['geocentric']
                jth_satellite_position = timeseries_footprints[j][t]['geocentric']
                dist = euclidean_distance(ith_satellite_position, jth_satellite_position)
                if dist <= possible_distance:
                    if i in graph:
                        graph[i].append(j)
                    else :
                        graph[i] = [j]
                    if j in graph:
                        graph[j].append(i)
                    else :
                        graph[j] = [i]
        timeseries_ISL_graph.append(graph)

    print(f'ISL possible distance : {possible_distance}')
    
    return timeseries_ISL_graph


def plot(timeseries_footprints):
    fig1 = plt.figure(1, figsize=(10, 5))
    fig1_ax = fig1.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    polygons = globalmap(fig1_ax, timeseries_footprints)
    timeseries_ISL_graph = make_timeseries_ISL_graph(timeseries_footprints)
    graph = []

    ani = animation.FuncAnimation(
        fig1, animate_util, len(timeseries_footprints[0]), 
        fargs=(
            timeseries_footprints,
            polygons,
            timeseries_ISL_graph,
            graph,
            fig1_ax
        ),
        interval=100
    )

    plt.show()


def main():
    t0 = ts.now() # get current time
    duration = timedelta(days=1)
    t1 = t0 + duration

    satellites = load_satellites('TLE/ISL_test.txt')
    print(f'Loaded {len(satellites)} satellites')

    print('Calculate Satellite Footprint...')
    timeseries_footprints = []
    
    for satellite in satellites.values():
        timeseries_footprints.append([
            get_footprint_at(t0 + timedelta(minutes=i), satellite)
            for i in range(int(duration.total_seconds()/60) + 1)
        ])
        print(satellite.name, '(finished)')

    plot(timeseries_footprints)
    
    
main()