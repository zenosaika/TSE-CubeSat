from skyfield.api import load, wgs84
from datetime import timedelta
import math
import numpy as np
import seaborn as sns
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from shapely import Polygon


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


def find_events(satellite, observer_latlon, t0, t1):
    times, events = satellite.find_events(observer_latlon, t0, t1, altitude_degrees=MINIMUM_ELEVATION_ANGLE)
    event_names = [
        'ดาวเทียมเข้ามาในพิสัย', 
        'อยู่ในจุด upper culmination', 
        'ดาวเทียมออกจากพิสัย'
    ]
    
    event_and_times = []

    print()
    print(f'<<< {satellite.name} >>>')
    for time, event in zip(times, events):
        name = event_names[event]
        event_and_times.append((['rise', 'culminate', 'set'][event], time))
        print(time.utc_strftime('%Y %b %d %H:%M:%S (UTC)'), name)

    return event_and_times


def get_information_at(t, satellite, observer_latlon):
    geocentric = satellite.at(t) # relative to center of the Earth

    lat, lon = wgs84.latlon_of(geocentric)

    difference = satellite - observer_latlon
    topocentric = difference.at(t) # relative to observer

    alt, az, distance = topocentric.altaz()

    return {
        'name': satellite.name,
        'datetime': t,
        'geocentric': geocentric.position.km,
        'topocentric': topocentric.position.km,
        'latitude': lat,
        'longitude': lon,
        'azimuth': az,
        'altitude': alt,
        'distance': distance,
        'visible': alt.degrees > MINIMUM_ELEVATION_ANGLE
    }


def geocentric_subplot(ax2, timeseries_informations):
    # plot center of the Earth at (0, 0, 0)
    ax2.plot(0, 0, 0, 'go')
    ax2.text(0, 0, 0, 'Center of Earth')

    geocentrics = []

    for timeseries_information in timeseries_informations:
        geocentrics.append(np.array([info['geocentric'] for info in timeseries_information]))

    ax2_dots = [ax2.plot([], [], [], marker='o', color='r')[0] for _ in geocentrics]
    ax2_lines = [ax2.plot([], [], [], color='b', alpha=0.7)[0] for _ in geocentrics]

    get_axis_range = lambda axis : (min(np.amin(arr2d[:, axis]) for arr2d in geocentrics), max(np.amax(arr2d[:, axis]) for arr2d in geocentrics))
    ax2.set(xlim3d=get_axis_range(0), xlabel='X (km)')
    ax2.set(ylim3d=get_axis_range(1), ylabel='Y (km)')
    ax2.set(zlim3d=get_axis_range(2), zlabel='Z (km)')

    # draw a sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = EARTH_RADIUS*np.cos(u)*np.sin(v)
    y = EARTH_RADIUS*np.sin(u)*np.sin(v)
    z = EARTH_RADIUS*np.cos(v)
    ax2.plot_surface(x, y, z, color="g", alpha=0.1)

    ax2.axis('equal') # set aspect ratio to equal

    return geocentrics, ax2_dots, ax2_lines


def timeslots_subplot(ax3, t0, satellites, events_of_each_satellite):
    ith_satellite = 1

    for events in events_of_each_satellite:
        xranges = []
        start_time = t0

        for event, time in events:
            if event == 'culminate':
                continue
            elif event == 'rise':
                start_time = time
            elif event == 'set':
                start_minute = timedelta(days=start_time - t0).total_seconds() / 60
                duration = timedelta(days=time - start_time).total_seconds() / 60
                xranges.append((start_minute, duration))
        
        ax3.broken_barh(xranges, (10*ith_satellite, 9), facecolors='tab:green')

        for each_slot in xranges:
            margin = 5
            x = sum(each_slot) + margin
            ax3.text(x, 10*ith_satellite, f'{each_slot[1]:.1f}m', fontsize=7)

        ith_satellite += 1

    ax3.set_yticks(
        [10*ith_satellite+5 for ith_satellite in range(1, len(satellites)+1)],
        labels=[satellite.name for satellite in satellites.values()]
    )

    ax3.set_xlim(0, 1440) # minutes in a day
    ax3.set_xlabel(f"[Minutes from {t0.utc_strftime('%Y %b %d %H:%M:%S (UTC)')}]")
    ax3.grid(True)

    ax3_vline = ax3.axvline(x=0, color='black')

    return ax3_vline


def polar_subplot(ax4, timeseries_informations):
    azimuths = []
    elevations = []
    for timeseries_information in timeseries_informations:
        azimuth = [info['azimuth'].dms()[0] * np.pi/180 for info in timeseries_information]
        elevation = [90 - info['altitude'].dms()[0] for info in timeseries_information]
        azimuths.append(azimuth)
        elevations.append(elevation)

    ax4_dots = [ax4.plot([], [], color='black', marker='D', markerfacecolor='limegreen')[0] for _ in azimuths]

    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1) # clockwise
    ax4.grid(True)

    ax4.set_yticks(range(0, 90+10, 10))
    yLabel = ['90°', '', '', '60°', '', '', '30°', '', '', '']
    ax4.set_yticklabels(yLabel)
    ax4.set_xlabel('Satellite Azimuth & Altitude (at GS)')
    ax4.tick_params(axis='y', colors='crimson')

    return azimuths, elevations, ax4_dots


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
    # ax.title.set_text('')

    polygons = [
        ax.plot([], [], color='blue', transform=ccrs.PlateCarree())[0] 
        for _ in timeseries_footprints
    ]

    GS = Line2D([0], [0], marker='o', markersize=10, markeredgecolor='w', markerfacecolor='r', linestyle='')
    FP = Line2D([0], [0], marker='o', markersize=15, markeredgecolor='b', markerfacecolor='w', linestyle='')
    ISL = Line2D([0], [0], color='y')
    SAT = Line2D([0], [0], marker='o', markersize=10, markeredgecolor='w', markerfacecolor='k', linestyle='')
    ax.legend([GS, SAT, FP, ISL], ['Ground Station', 'Satellite', 'Satellite Footprint', 'Inter-satellite Link'], loc='lower left')

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

    ax.title.set_text('ROI Coverage in a day (%)')
    ax.set_xlabel(f"[Minutes from {start_time.utc_strftime('%Y %b %d %H:%M:%S (UTC)')}]")

    ax.axhline(y=0, color='k',linewidth=1)
    ax.axhline(y=1, color='k',linewidth=1)
    ax.axvline(x=0, color='k',linewidth=1)
    ax.axvline(x=len(percent_coverages), color='k',linewidth=1)


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
    possible_distance = 10**((Ls - 20*math.log10(f) + 147.55) / 20) # in meters

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


def animate_util(point_of_time_i, 
                timeseries_footprints, polygons, 
                timeseries_ISL_graph, graph, ax1, 
                ax2_dots, ax2_lines, geocentrics,
                ax3_vline,
                azimuths, elevations, ax4_dots):
    
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
            graph.append(ax1.plot([src_lon, dst_lon], [src_lat, dst_lat], color='y', marker='o', mfc='k', mec='w', transform=ccrs.PlateCarree())[0])

    # for subplot 2
    for ax2_dot, ax2_line, geocentric in zip(ax2_dots, ax2_lines, geocentrics):
        ax2_dot.set_data(geocentric[point_of_time_i, 0], geocentric[point_of_time_i, 1])
        ax2_dot.set_3d_properties(geocentric[point_of_time_i, 2])

        ax2_line.set_data(geocentric[:point_of_time_i, 0], geocentric[:point_of_time_i, 1])
        ax2_line.set_3d_properties(geocentric[:point_of_time_i, 2])

    # for subplot 3
    ax3_vline.set_data([point_of_time_i, point_of_time_i], [0, 1])

    # for subplot 4
    for ax4_dot, azimuth, elevation in zip(ax4_dots, azimuths, elevations):
        ax4_dot.set_data(azimuth[point_of_time_i], elevation[point_of_time_i])


def plot(timeseries_footprints, preferred_region, timeseries_informations, events_of_each_satellite, t0, satellites, observer_latlon):
    fig = plt.figure(figsize=(15, 8))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=0.93, wspace=0.2, hspace=0.4)
    ax1 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2, projection=ccrs.Robinson())
    ax2 = plt.subplot2grid((3, 3), (1, 2), rowspan=2, projection='3d')
    ax3 = plt.subplot2grid((3, 3), (0, 0))
    ax4 = plt.subplot2grid((3, 3), (0, 1))
    ax4.title.set_text('Uplink / Downlink Slots in a day (at GS)')
    ax5 = plt.subplot2grid((3, 3), (0, 2), projection='polar')

    polygons = globalmap(ax1, timeseries_footprints)
    lons = [float(p[0]) for p in preferred_region]
    lats = [float(p[1]) for p in preferred_region]
    ax1.plot(lons, lats, color='k', transform=ccrs.PlateCarree()) # draw ROI (Thailand)
    ax1.plot(observer_latlon.longitude.degrees, observer_latlon.latitude.degrees, marker='o', mfc='r', mec='w', transform=ccrs.PlateCarree()) # draw observer (TSE Building)

    timeseries_ISL_graph = make_timeseries_ISL_graph(timeseries_footprints)
    graph = []

    percent_coverages = get_percent_coverages(timeseries_footprints, preferred_region)
    start_time = timeseries_footprints[0][0]['datetime']
    heatmap(ax3, percent_coverages, start_time)

    geocentrics, ax2_dots, ax2_lines = geocentric_subplot(ax2, timeseries_informations)
    ax4_vline = timeslots_subplot(ax4, t0, satellites, events_of_each_satellite)
    azimuths, elevations, ax5_dots = polar_subplot(ax5, timeseries_informations)

    ani = animation.FuncAnimation(
        fig, animate_util, len(timeseries_footprints[0]), 
        fargs=(
            timeseries_footprints, polygons,
            timeseries_ISL_graph, graph, ax1,
            ax2_dots, ax2_lines, geocentrics,
            ax4_vline,
            azimuths, elevations, ax5_dots,
        ),
        interval=100,
    )
    
    # fig.tight_layout()
    plt.show()


def main():
    observer_latlon = wgs84.latlon(+14.0691107, +100.6051873) # at TSE Building

    t0 = ts.now() # get current time
    duration = timedelta(days=1)
    t1 = t0 + duration
    
    with open('polygon/thailand_polygon.txt', 'r') as f:
        thailand = [line.split(', ') for line in f.readlines() if line!='']

    satellites = load_satellites('TLE/ISL_test.txt')
    print(f'Loaded {len(satellites)} satellites')

    print('Calculate Satellite Data...')
    timeseries_footprints = []
    timeseries_informations = []
    events_of_each_satellite = []
    
    for satellite in satellites.values():
        timeseries_footprints.append([
            get_footprint_at(t0 + timedelta(minutes=i), satellite)
            for i in range(int(duration.total_seconds()/60) + 1)
        ])

        timeseries_information = [
            get_information_at(t0 + timedelta(minutes=i), satellite, observer_latlon)
            for i in range(24*60+1)
        ]
        timeseries_informations.append(timeseries_information)

        events = find_events(satellite, observer_latlon, t0, t1)
        events_of_each_satellite.append(events)

        print(satellite.name, '(finished)')

    plot(timeseries_footprints, thailand, timeseries_informations, events_of_each_satellite, t0, satellites, observer_latlon)
    
    
main()