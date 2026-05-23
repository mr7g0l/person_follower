#!/bin/bash
# ============================================================================
# record_experiment.bash - records a rosbag2 of a person-follower experiment.
#
# Usage:
#   ./record_experiment.bash [name]
#
# Creates a rosbag under  report/rosbags/<name>_<timestamp>  containing the
# topics needed to (a) plot the trajectories and (b) replay the experiment.
# Stop the recording with Ctrl-C.
# ============================================================================
set -e

source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

NAME="${1:-exp}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${REPO_DIR}/report/rosbags/${NAME}_${STAMP}"

# Lightweight topics: trajectory analysis + experiment replay.
# (Camera/depth image topics are intentionally omitted - they make the bag
#  very large. Add them below if you need to replay the vision pipeline.)
TOPICS=(
  /odom
  /cmd_vel
  /scan
  /person_follower/target_estimate
  /person_follower/target_measurement
)

mkdir -p "${REPO_DIR}/report/rosbags"
echo "Recording rosbag : ${OUTDIR}"
echo "Topics           : ${TOPICS[*]}"
echo "Press Ctrl-C to stop."
echo ""
ros2 bag record -o "${OUTDIR}" "${TOPICS[@]}"
