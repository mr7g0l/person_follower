#!/usr/bin/env python3
"""
plot_trajectories.py - XY trajectory plots for the person-follower experiments.

It overlays, in the Webots world frame:
  * the robot trajectory                 (from /odom in the rosbag)
  * the perception measurement / Kalman  (from /person_follower/target_*)
  * the ground-truth pedestrian paths    (waypoints parsed from the .wbt world)

Three figures are produced: a global overview, a tracking-detail plot and a
robot-to-target distance plot.

USAGE
-----
  # From a recorded rosbag (run with ROS 2 sourced):
  python3 plot_trajectories.py --bag rosbags/exp_01

  # Demonstration mode (synthetic data, needs no ROS / no rosbag) - used to
  # produce the template figures embedded in the report:
  python3 plot_trajectories.py --demo

The robot's /odom data is expressed in the 'odom' frame, whose origin is the
robot start pose.  This script reads that start pose from the .wbt world file
and transforms the odometry into Webots world coordinates so it can be drawn
together with the pedestrian waypoints.
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORLD = os.path.normpath(
    os.path.join(HERE, '..', 'webots',
                 'turtlebot3_burger_pedestrian_simple.wbt'))

# ArUco marker world positions (x, y), from person_follower.py
ARUCO_MARKERS = {0: (7.5, 5.0), 1: (8.0, -5.0), 2: (10.0, 2.0), 3: (10.0, -2.0)}

# Topics recorded for the trajectory analysis
TOPICS = {
    '/odom': 'robot',
    '/person_follower/target_estimate': 'estimate',
    '/person_follower/target_measurement': 'measurement',
}


# ------------------------- Webots world parser -----------------------------
def parse_world(path):
    """Return (robot_start, pedestrian_waypoints, pedestrian_speeds)."""
    with open(path, 'r') as f:
        text = f.read()

    robot = {'pos': (0.0, 0.0), 'yaw': 0.0}
    idx = text.find('TurtleBot3Burger {')
    if idx >= 0:
        seg = text[idx:idx + 400]
        mt = re.search(
            r'translation\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)', seg)
        if mt:
            robot['pos'] = (float(mt.group(1)), float(mt.group(2)))
        mr = re.search(
            r'rotation\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)',
            seg)
        if mr:
            _, _, az, ang = (float(v) for v in mr.groups())
            robot['yaw'] = ang if az >= 0 else -ang

    peds = []
    for m in re.finditer(r'--trajectory=([^"]+)', text):
        pts = []
        for pair in m.group(1).split(','):
            xy = pair.split()
            if len(xy) >= 2:
                try:
                    pts.append((float(xy[0]), float(xy[1])))
                except ValueError:
                    pass
        if len(pts) >= 2:
            peds.append(pts)

    speeds = [float(s) for s in re.findall(r'--speed=([\d.]+)', text)]
    return robot, peds, speeds


# ------------------------- Geometry helpers --------------------------------
def odom_to_world(pts, robot):
    """Transform (x, y) points from the 'odom' frame to the Webots world frame."""
    th = robot['yaw']
    c, s = math.cos(th), math.sin(th)
    ox, oy = robot['pos']
    return [(ox + c * x - s * y, oy + s * x + c * y) for (x, y) in pts]


def densify(waypoints, step=0.04):
    """Resample a polyline so consecutive points are ~step metres apart."""
    out = []
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg / step))
        for i in range(n):
            t = i / n
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    out.append(waypoints[-1])
    return out


def smooth(arr, win):
    """Moving-average smoothing along axis 0 with edge padding (no boundary
    drift)."""
    arr = np.asarray(arr, dtype=float)
    if win < 2 or len(arr) < win:
        return arr
    if win % 2 == 0:
        win += 1
    half = win // 2
    k = np.ones(win) / win
    out = np.empty_like(arr)
    for c in range(arr.shape[1]):
        padded = np.pad(arr[:, c], half, mode='edge')
        out[:, c] = np.convolve(padded, k, mode='valid')
    return out


def path_length(pts):
    """Total arc length of a list of (t, x, y) samples."""
    if len(pts) < 2:
        return 0.0
    p = np.array([(x, y) for _, x, y in pts])
    return float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))


# ------------------------- rosbag2 reader ----------------------------------
def read_bag(bag_path, storage_id):
    """Read /odom and the target topics from a rosbag2 directory."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        sys.exit("ERROR: the ROS 2 Python API is not available (%s).\n"
                 "Run 'source /opt/ros/humble/setup.bash' before using --bag."
                 % exc)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions('', ''))
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    data = {v: [] for v in TOPICS.values()}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        key = TOPICS.get(topic)
        if key is None:
            continue
        msg = deserialize_message(raw, get_message(type_map[topic]))
        ts = stamp * 1e-9
        if key == 'robot':
            p = msg.pose.pose.position
        else:
            p = msg.point
        data[key].append((ts, p.x, p.y))

    if not any(data.values()):
        sys.exit("ERROR: the rosbag contains none of the expected topics: %s"
                 % ', '.join(TOPICS))
    return data


# ------------------------- Demo (synthetic) data ---------------------------
def make_demo(robot, peds):
    """Build plausible synthetic data in WORLD coordinates for the template
    figures.  Replace with a real --bag run before submitting the report."""
    rng = np.random.default_rng(42)
    step = 0.04
    target = densify(peds[0], step=step)             # followed pedestrian
    n = len(target)
    speed = 0.15
    times = np.arange(n) * (step / speed)

    # Robot: simple pursuit that trails the target at ~0.8 m
    pos = np.array(robot['pos'], dtype=float)
    max_step = 0.055                                 # slightly faster than target
    robot_pts = np.empty((n, 2))
    for i in range(n):
        to_tgt = np.array(target[i]) - pos
        dist = math.hypot(to_tgt[0], to_tgt[1])
        if dist > 1e-6:
            move = float(np.clip(dist - 0.8, -max_step, max_step))
            pos = pos + (to_tgt / dist) * move
        robot_pts[i] = pos
    robot_pts += rng.normal(0, 0.02, robot_pts.shape)
    robot_pts = smooth(robot_pts, 11)

    meas = np.array(target) + rng.normal(0, 0.12, (n, 2))
    est = smooth(meas, 25)

    return {
        'robot':       [(times[i], robot_pts[i, 0], robot_pts[i, 1])
                        for i in range(n)],
        'measurement': [(times[i], meas[i, 0], meas[i, 1]) for i in range(n)],
        'estimate':    [(times[i], est[i, 0], est[i, 1]) for i in range(n)],
    }


# ------------------------- Distance vs. time -------------------------------
def distance_series(robot, estimate):
    """Robot-to-target distance resampled onto a common time grid."""
    if len(robot) < 2 or len(estimate) < 2:
        return None, None
    tr = np.array([p[0] for p in robot])
    rx = np.array([p[1] for p in robot])
    ry = np.array([p[2] for p in robot])
    te = np.array([p[0] for p in estimate])
    ex = np.array([p[1] for p in estimate])
    ey = np.array([p[2] for p in estimate])
    t0, t1 = max(tr[0], te[0]), min(tr[-1], te[-1])
    if t1 <= t0:
        return None, None
    grid = np.linspace(t0, t1, 400)
    d = np.hypot(np.interp(grid, te, ex) - np.interp(grid, tr, rx),
                 np.interp(grid, te, ey) - np.interp(grid, tr, ry))
    return grid - grid[0], d


# ------------------------- Plotting ----------------------------------------
def _xy(samples):
    return ([s[1] for s in samples], [s[2] for s in samples])


def plot_overview(data, peds, robot, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    ped_styles = [('Pedestrian 1 (target, ground truth)', '#c0392b'),
                  ('Pedestrian 2 (ground truth)', '#e67e22')]
    for i, ped in enumerate(peds):
        label, color = ped_styles[i] if i < len(ped_styles) else (
            'Pedestrian %d (ground truth)' % (i + 1), '#7f8c8d')
        xs = [p[0] for p in ped]
        ys = [p[1] for p in ped]
        ax.plot(xs, ys, '--', color=color, lw=2, label=label)
        ax.plot(xs[0], ys[0], 'o', color=color, ms=9, mec='k', zorder=5)

    if data.get('estimate'):
        ex, ey = _xy(data['estimate'])
        ax.plot(ex, ey, '-', color='#27ae60', lw=1.8,
                label='Target - Kalman estimate')
    if data.get('robot'):
        rx, ry = _xy(data['robot'])
        ax.plot(rx, ry, '-', color='#2c3e50', lw=2.4, label='Robot trajectory')
        ax.plot(rx[0], ry[0], 's', color='#2c3e50', ms=11, mec='k',
                label='Robot start', zorder=6)
        ax.plot(rx[-1], ry[-1], '*', color='#2c3e50', ms=18, mec='k',
                label='Robot end', zorder=6)

    for mid, (mx, my) in ARUCO_MARKERS.items():
        ax.plot(mx, my, 's', color='k', ms=10)
        ax.annotate('ArUco %d' % mid, (mx, my), textcoords='offset points',
                    xytext=(8, 4), fontsize=8)

    ax.set_xlabel('World X [m]')
    ax.set_ylabel('World Y [m]')
    ax.set_title('Experiment overview - robot and pedestrian trajectories')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='best', fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('  wrote', out_path)


def plot_following(data, peds, out_path):
    fig, ax = plt.subplots(figsize=(9, 6.5))

    target = peds[0]
    ax.plot([p[0] for p in target], [p[1] for p in target], '--',
            color='#c0392b', lw=2, label='Pedestrian 1 (ground truth)')

    if data.get('measurement'):
        mx, my = _xy(data['measurement'])
        ax.scatter(mx, my, s=10, color='#95a5a6', alpha=0.5,
                   label='Fused measurement (depth + LiDAR)')
    if data.get('estimate'):
        ex, ey = _xy(data['estimate'])
        ax.plot(ex, ey, '-', color='#27ae60', lw=2, label='Kalman estimate')
    if data.get('robot'):
        rx, ry = _xy(data['robot'])
        ax.plot(rx, ry, '-', color='#2c3e50', lw=2.4, label='Robot trajectory')

    ax.set_xlabel('World X [m]')
    ax.set_ylabel('World Y [m]')
    ax.set_title('Target tracking detail - measurement, Kalman estimate, robot')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='best', fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('  wrote', out_path)


def plot_distance(data, out_path, desired=0.8):
    t, d = distance_series(data.get('robot', []), data.get('estimate', []))
    if t is None:
        print('  (skipped distance plot - not enough data)')
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, d, '-', color='#2980b9', lw=1.8, label='Robot-to-target distance')
    ax.axhline(desired, color='#c0392b', ls='--', lw=1.5,
               label='Desired distance (%.2f m)' % desired)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance [m]')
    ax.set_title('Following distance over time')
    ax.grid(True, ls=':', alpha=0.6)
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('  wrote', out_path)


# ------------------------- Main --------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='XY trajectory plots for the person-follower experiments.')
    ap.add_argument('--bag', help='path to a rosbag2 directory')
    ap.add_argument('--demo', action='store_true',
                    help='generate synthetic template figures (no ROS needed)')
    ap.add_argument('--world', default=DEFAULT_WORLD,
                    help='Webots .wbt world file (default: simple world)')
    ap.add_argument('--out', default=HERE,
                    help='output directory for the PNG figures')
    ap.add_argument('--storage', default='sqlite3',
                    help='rosbag2 storage id (sqlite3 or mcap)')
    args = ap.parse_args()

    if not args.bag and not args.demo:
        ap.error('provide either --bag <dir> or --demo')

    robot, peds, speeds = parse_world(args.world)
    print('World : %s' % args.world)
    print('Robot start : pos=(%.2f, %.2f)  yaw=%.3f rad'
          % (robot['pos'][0], robot['pos'][1], robot['yaw']))
    print('Pedestrians : %d trajectories parsed' % len(peds))
    if not peds:
        sys.exit('ERROR: no pedestrian trajectories found in the world file.')

    if args.demo:
        print('Mode  : DEMO (synthetic data - replace with a real rosbag)')
        data = make_demo(robot, peds)
    else:
        print('Mode  : rosbag  (%s)' % args.bag)
        data = read_bag(args.bag, args.storage)
        # /odom and target topics are in the 'odom' frame -> world frame
        for key in list(data):
            if data[key]:
                ts = [s[0] for s in data[key]]
                xy = odom_to_world([(s[1], s[2]) for s in data[key]], robot)
                data[key] = [(ts[i], xy[i][0], xy[i][1])
                             for i in range(len(ts))]

    os.makedirs(args.out, exist_ok=True)
    plot_overview(data, peds, robot, os.path.join(args.out, 'traj_overview.png'))
    plot_following(data, peds, os.path.join(args.out, 'traj_following.png'))
    plot_distance(data, os.path.join(args.out, 'traj_distance.png'))

    # -- Summary statistics (useful to quote real numbers in the report) --
    print('')
    print('Summary')
    print('  robot path length   : %6.2f m'
          % path_length(data.get('robot', [])))
    print('  target path length  : %6.2f m'
          % path_length(data.get('estimate', [])))
    t, d = distance_series(data.get('robot', []), data.get('estimate', []))
    if t is not None:
        print('  following distance  : min %.2f / mean %.2f / max %.2f m'
              % (d.min(), d.mean(), d.max()))
    print('Done.')


if __name__ == '__main__':
    main()
