# Person Follower — ROS 2 / Webots

A person-following robot for the TurtleBot3 Burger, running in a Webots
simulation and controlled by a ROS 2 Humble node. The robot detects a target
pedestrian and follows it by fusing four sensing modalities — RGB camera
(YOLOv8 + BoT-SORT), depth camera, LiDAR and odometry — with a 2-D Kalman
filter for motion prediction and ArUco markers as a localisation backup.

The full technical documentation is in
[`report/person_follower_report.pdf`](report/person_follower_report.pdf).

## Requirements

- Ubuntu 22.04
- [ROS 2 Humble](https://docs.ros.org/en/humble/index.html)
- [Webots R2025a](https://github.com/cyberbotics/webots/releases/tag/R2025a),
  extracted to `$HOME/webots-R2025a`

## 1. Install prerequisites

```
sudo apt update
sudo apt install ros-humble-webots-ros2-turtlebot ros-humble-cv-bridge
pip3 install ultralytics opencv-contrib-python numpy matplotlib
```

## 2. Build the package

```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/mr7g0l/person_follower.git
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

## 3. Generate the ArUco textures (optional)

The marker PNGs are already included in the repository. To regenerate them:

```
cd ~/ros2_ws/src/person_follower
python3 webots/generate_aruco_markers.py
```

## 4. Install the world files

Copy the world files **and** the ArUco textures into the
`webots_ros2_turtlebot` package, so that Webots can locate the texture images
referenced by the world:

```
WORLDS=/opt/ros/humble/share/webots_ros2_turtlebot/worlds
sudo cp ~/ros2_ws/src/person_follower/webots/*.wbt       $WORLDS/
sudo cp ~/ros2_ws/src/person_follower/webots/aruco_*.png $WORLDS/
```

## 5. Run the simulation

Use three terminals and source ROS 2 in each one.

**Terminal 1 — person-follower node**

```
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=1
ros2 run person_follower person_follower
```

**Terminal 2 — Webots simulator**

```
export WEBOTS_HOME=~/webots-R2025a
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1
ros2 launch webots_ros2_turtlebot robot_launch.py \
  world:=turtlebot3_burger_pedestrian_simple.wbt
```

A room without walls is also available:
`world:=turtlebot3_burger_pedestrian_no_walls.wbt`

**Terminal 3 — RViz (optional)**

```
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1
rviz2 -d ~/ros2_ws/src/person_follower/webots/config.rviz
```

Once Webots is running the robot locks onto the nearest pedestrian and starts
following it. The annotated camera stream is published on
`/person_follower/debug_image`.

## 6. Record an experiment (rosbag)

With the simulation running, start a recording in a new terminal:

```
cd ~/ros2_ws/src/person_follower/webots
./record_experiment.bash exp_01
```

This records `/odom`, `/cmd_vel`, `/scan`,
`/person_follower/target_measurement` and `/person_follower/target_estimate`
into `report/rosbags/exp_01_<timestamp>`. Press `Ctrl-C` to stop.

Replay a recording with:

```
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1
ros2 bag play report/rosbags/exp_01_<timestamp>
```

## 7. Plot the trajectories

```
cd ~/ros2_ws/src/person_follower/report
python3 plot_trajectories.py --bag rosbags/exp_01_<timestamp>
```

This generates `traj_overview.png`, `traj_following.png` and
`traj_distance.png` and prints the summary statistics of the run. Use `--demo`
instead of `--bag` to produce example figures without a rosbag.

## Recorded rosbags

The experiment rosbags are too large for Git (`report/rosbags/` is excluded in
`.gitignore`) and are shared via Google Drive:

[Recorded rosbags - Google Drive](https://drive.google.com/drive/folders/1HjiBTtxrNCbCuHkJ0b46jUU2g8fwfBa-?usp=drive_link)

Download a bag and use it with `ros2 bag play` or `plot_trajectories.py` as
described above.

## Repository layout

```
person_follower/   ROS 2 node (perception + control)
webots/            Webots worlds, ArUco generator, rosbag recording script
report/            LaTeX report, trajectory plotting script, figures
```

## Documentation

See [`report/person_follower_report.pdf`](report/person_follower_report.pdf)
for the full design and implementation report, including the algorithm
references, the trajectory analysis and the complete run instructions.
