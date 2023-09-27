from skyfield.api import load, wgs84
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    
    print()
    for time, event in zip(times, events):
        name = event_names[event]
        print(time.utc_strftime('%Y %b %d %H:%M:%S (UTC)'), name)


def get_information_at(t, satellite, observer_latlon):
    geocentric = satellite.at(t) # relative to center of the Earth

    lat, lon = wgs84.latlon_of(geocentric)

    difference = satellite - observer_latlon
    topocentric = difference.at(t) # relative to observer

    alt, az, distance = topocentric.altaz()

    return {
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


def animate_orbit(point_of_time_i, lines, positions, is_visibles):
    for line, position, is_visible in zip(lines, positions, is_visibles):
        line.set_data(position[point_of_time_i, :2].T)
        line.set_3d_properties(position[point_of_time_i, 2])
        line.set_color('g' if is_visible[point_of_time_i] else 'r')
    return lines


def animate(timeseries_information):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot observer at (0, 0, 0)
    ax.plot(0, 0, 0, 'bo')
    ax.text(0, 0, 0, 'Observer')
    
    positions = []
    is_visibles = []

    this_satellite_position = np.array([info['topocentric'] for info in timeseries_information])
    positions.append(this_satellite_position)

    this_satellite_visible_flags = [info['visible'] for info in timeseries_information]
    is_visibles.append(this_satellite_visible_flags)

    lines = [ax.plot([], [], [], marker='o', color='r')[0] for _ in positions]

    plt.title('')
    get_axis_range = lambda axis : (min(np.amin(arr2d[:, axis]) for arr2d in positions), max(np.amax(arr2d[:, axis]) for arr2d in positions))
    ax.set(xlim3d=get_axis_range(0), xlabel='X (km)')
    ax.set(ylim3d=get_axis_range(1), ylabel='Y (km)')
    ax.set(zlim3d=get_axis_range(2), zlabel='Z (km)')

    # draw a plane
    # xx, zz = np.meshgrid(np.linspace(-7000, 7000, 100), np.linspace(-7000, 7000, 100))
    # yy = zz * 0
    # ax.plot_surface(xx, yy, zz, color="y", alpha=0.1)

    # draw a sphere
    r = 5000
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    ax.plot_surface(x, y, z, color="y", alpha=0.1)

    ani = animation.FuncAnimation(fig, animate_orbit, len(timeseries_information), fargs=(lines, positions, is_visibles), interval=100)
    # ani.save(filename='animation')

    ax.axis('equal') # set aspect ratio to equal
    plt.show()


def main():
    observer_latlon = wgs84.latlon(+18.3170581, +99.3986862) # at Thammasat
    t0 = ts.now() # get current time
    t1 = t0 + timedelta(days=1)

    satellites = load_satellites('TLE/THEOS1.txt')
    satellite = satellites['THEOS'] # set preferred satellite

    find_events(satellite, observer_latlon, t0, t1)

    timeseries_information = [
        get_information_at(t0 + timedelta(minutes=i), satellite, observer_latlon)
        for i in range(24*60+1)
    ]

    animate(timeseries_information)

    
main()