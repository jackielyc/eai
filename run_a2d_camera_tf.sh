#!/usr/bin/env bash
# Publish base_link -> camera_frame for A2D head camera.
# Run inside the robot ROS container (same ROS_DOMAIN_ID as the robot).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${ROS_DOMAIN_ID:=0}"
export ROS_DOMAIN_ID

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/psi/rt/a2d-tele/install/setup.bash
fi

exec python3.10 "${SCRIPT_DIR}/a2d_head_camera_tf.py" "$@"
