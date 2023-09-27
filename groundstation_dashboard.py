from skyfield.api import load, wgs84
from datetime import timedelta
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

mpl.font_manager.fontManager.addfont('THSarabunChula-Regular.ttf')
font = {'family' : 'TH Sarabun Chula',
        'weight' : 'bold',
        'size'   : 16}
mpl.rc('font', **font)

MINIMUM_ELEVATION_ANGLE = 10.0 # should be dynamic (will update in lv.4)

ts = load.timescale()


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


def animate_util(point_of_time_i, ax1_dots, topocentrics, is_visibles,
                 ax2_dots, ax2_lines, geocentrics,
                 ax3_vline,
                 azimuths, elevations, ax4_dots):
    # for subplot 1
    for ax1_dot, topocentric, is_visible in zip(ax1_dots, topocentrics, is_visibles):
        ax1_dot.set_data(topocentric[point_of_time_i, 0], topocentric[point_of_time_i, 1])
        ax1_dot.set_3d_properties(topocentric[point_of_time_i, 2])
        ax1_dot.set_color('g' if is_visible[point_of_time_i] else 'r')

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


def topocentric_subplot(ax1, timeseries_informations):
    # plot observer at (0, 0, 0)
    ax1.plot(0, 0, 0, 'bo')
    ax1.text(0, 0, 0, 'Observer')
    
    topocentrics = []
    is_visibles = []

    for timeseries_information in timeseries_informations:
        topocentrics.append(np.array([info['topocentric'] for info in timeseries_information]))
        is_visibles.append([info['visible'] for info in timeseries_information])

    ax1_dots = [ax1.plot([], [], [], marker='o', color='r')[0] for _ in topocentrics]

    get_axis_range = lambda axis : (min(np.amin(arr2d[:, axis]) for arr2d in topocentrics), max(np.amax(arr2d[:, axis]) for arr2d in topocentrics))
    ax1.set(xlim3d=get_axis_range(0), xlabel='X (km)')
    ax1.set(ylim3d=get_axis_range(1), ylabel='Y (km)')
    ax1.set(zlim3d=get_axis_range(2), zlabel='Z (km)')

    return topocentrics, is_visibles, ax1_dots


def geocentric_subplot(ax2, timeseries_informations):
    # plot center of the Earth at (0, 0, 0)
    ax2.plot(0, 0, 0, 'go')
    ax2.text(0, 0, 0, 'Earth')

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
    r = 6378 # Earth's radius
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
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
    ax4.tick_params(axis='y', colors='crimson')

    return azimuths, elevations, ax4_dots


def animation_plot(timeseries_informations, events_of_each_satellite, t0, satellites):
    fig = plt.figure(1, figsize=(13, 8))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.4)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.title.set_text('Topocentric (Relatives to Observer)')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.title.set_text('Geocentric (Relatives to the Earth)')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.title.set_text('Time Slots for Uplink / Downlink in a day')
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    ax4.title.set_text('Azimuth / Elevation angle of Satellites')


    topocentrics, is_visibles, ax1_dots = topocentric_subplot(ax1, timeseries_informations)
    geocentrics, ax2_dots, ax2_lines = geocentric_subplot(ax2, timeseries_informations)
    ax3_vline = timeslots_subplot(ax3, t0, satellites, events_of_each_satellite)
    azimuths, elevations, ax4_dots = polar_subplot(ax4, timeseries_informations)

    ani = animation.FuncAnimation(fig, animate_util, len(timeseries_informations[0]), 
                                  fargs=(ax1_dots, topocentrics, is_visibles,
                                         ax2_dots, ax2_lines, geocentrics,
                                         ax3_vline,
                                         azimuths, elevations, ax4_dots,
                                        ),
                                  interval=100)
    # ani.save(filename='animation')

    plt.show()


def visible_event(timeseries_informations, t0):
    freq_array = [0 for _ in range(len(timeseries_informations[0]))]

    for timeseries_information in timeseries_informations:
        for i, info in enumerate(timeseries_information):
            if info['visible']:
                freq_array[i] += 1

    x = 0.5 + np.arange(len(freq_array))
    y = freq_array

    fig = plt.figure(2, figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x, y, width=1, edgecolor="black", linewidth=0.7)
    ax1.title.set_text('')
    ax1.set_xlabel(f"[Minutes from {t0.utc_strftime('%Y %b %d %H:%M:%S (UTC)')}]")
    ax1.set_ylabel('Number of Visible Satellites')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(x, y, width=1, edgecolor="black", linewidth=0.7)
    ax2.title.set_text('')
    ax2.set_xlabel(f"[Minutes from {t0.utc_strftime('%Y %b %d %H:%M:%S (UTC)')}]")
    ax2.set_ylabel('Number of Visible Satellites')


def extract_events(events_of_each_satellite, timeseries_informations, t0):
    event_names = {
        'rise': 'ดาวเทียมเข้ามาในพิสัย', 
        'culminate': 'อยู่ในจุด upper culmination', 
        'set': 'ดาวเทียมออกจากพิสัย'
    }

    unsorted_events = []

    for events, timeseries_information in zip(events_of_each_satellite, timeseries_informations):
        for event in events:
            satellite_name = timeseries_information[0]['name']
            event_name = event_names[event[0]]
            time = event[1]
            minutes_since_t0 = int(timedelta(days=time - t0).total_seconds() / 60)
            azimuth = timeseries_information[minutes_since_t0]['azimuth']
            elevation = timeseries_information[minutes_since_t0]['altitude']
            unsorted_events.append((minutes_since_t0, satellite_name, event_name, time, azimuth, elevation))

    return sorted(unsorted_events)


def main():
    observer_latlon = wgs84.latlon(+14.0691107, +100.6051873) # at TSE Building
    t0 = ts.now() # get current time
    t1 = t0 + timedelta(days=1)

    satellites = load_satellites('TLE/test.txt')
    print(f'Loaded {len(satellites)} satellites')

    timeseries_informations = []
    events_of_each_satellite = []
    
    for satellite in satellites.values():
        timeseries_information = [
            get_information_at(t0 + timedelta(minutes=i), satellite, observer_latlon)
            for i in range(24*60+1)
        ]
        timeseries_informations.append(timeseries_information)
        events = find_events(satellite, observer_latlon, t0, t1)
        events_of_each_satellite.append(events)

    print()
    sorted_events = extract_events(events_of_each_satellite, timeseries_informations, t0)
    for minutes_since_t0, satellite_name, event_name, time, azimuth, elevation in sorted_events:
        print(f"[{satellite_name}] {time.utc_strftime('%Y %b %d %H:%M:%S (UTC)')} {event_name} ที่ Azimuth: {azimuth}, Elevation: {elevation}")

    visible_event(timeseries_informations, t0)
    animation_plot(timeseries_informations, events_of_each_satellite, t0, satellites)

    
main()