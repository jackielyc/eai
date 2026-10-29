#!/usr/bin/env bash
set -eo pipefail
if [[ -f /opt/ros/humble/setup.bash ]]; then source /opt/ros/humble/setup.bash; fi
for setup in "${EAI_GENIE_SETUP:-}" /home/psibot/Projects/rcl/robot-control-library/install/setup.bash /home/psibot/Projects/synrobot/install/setup.bash; do
    if [[ -n "$setup" && -f "$setup" ]]; then source "$setup"; break; fi
done
for genie in "${EAI_GENIE_PREFIX:-}" /home/psibot/a2d_sdk/forwarder/app; do
    if [[ -n "$genie" && -d "$genie/local/lib/python3.10/dist-packages/genie_msgs" ]]; then
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$genie/local/lib/python3.10/dist-packages"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$genie/lib"
        export AMENT_PREFIX_PATH="$genie:${AMENT_PREFIX_PATH:-}"
        break
    fi
done
prefix="$(dirname "$(dirname "$1")")"
for cmeel in "$prefix"/lib/python*/site-packages/cmeel.prefix; do
    if [[ -d "$cmeel" ]]; then
        export LD_LIBRARY_PATH="$cmeel/lib:${LD_LIBRARY_PATH:-}"
        for packages in "$cmeel"/lib/python*/site-packages; do
            export PYTHONPATH="$packages:${PYTHONPATH:-}"
        done
    fi
done
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$root/.cache/ros_execution_logs}"
mkdir -p "$ROS_LOG_DIR"
if [[ "$2" == --check-environment || "$2" == --probe-feedback ]]; then exec "$1" "$root/compat/grasp_execution_service.py" "$2"; fi
exec "$1" "$root/compat/grasp_execution_service.py" --directory "$2"
