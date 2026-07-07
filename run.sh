#!/usr/bin/env bash
# 使用系统 Python 3.10 在宿主机运行
#
# ⚠️  若 ROS/camera 在 Docker 内运行，宿主机无法收到 SHM 图像数据！
#     请改用: bash run_in_docker.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/usr/bin/python3.10
HOST_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_host.xml"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "错误: 未找到 ROS2 Humble，请先安装 ros-humble-rclpy 等包" >&2
    exit 1
fi

# shellcheck disable=SC1091
set +u
source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=1

if [[ -f "${HOST_DDS_XML}" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${HOST_DDS_XML}"
    echo ">>> 宿主机 DDS: ${HOST_DDS_XML} (SHM + UDP)"
else
    echo ">>> 使用默认 DDS 配置"
fi
echo ">>> ROS_DOMAIN_ID=0  ROS_LOCALHOST_ONLY=1"

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true
set -e

if ! "${PYTHON}" - <<'PY' 2>/dev/null; then
import numpy
import rclpy
from cv_bridge import CvBridge
assert numpy.__version__.startswith("1.")
PY
    echo "错误: 依赖未就绪。请先执行: bash install.sh" >&2
    exit 1
fi

topic_count=$("${PYTHON}" - <<'PY' 2>/dev/null || echo 0
import rclpy
rclpy.init()
from rclpy.node import Node
n = Node("preflight_check")
print(len(dict(n.get_topic_names_and_types())))
n.destroy_node()
rclpy.shutdown()
PY
)
camera_hz=$({ timeout 3 ros2 topic hz /camera/head_color 2>&1 | grep -c "average rate" || true; })
echo ">>> 预检: 发现 ${topic_count} 个 topic"

if [[ "${topic_count}" -ge 10 && "${camera_hz}" -eq 0 ]]; then
    echo ""
    echo ">>> ⚠️  检测到 topic 但无图像数据 — ROS 很可能在 Docker 内运行" >&2
    echo ">>>     Docker 使用 SHM 共享内存，宿主机无法接收图像" >&2
    echo ">>>     请改用: bash run_in_docker.sh" >&2
    echo ""
fi

unset QT_PLUGIN_PATH
QT5_PLATFORMS=$("${PYTHON}" -c "import PyQt5, os; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms'))")
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT5_PLATFORMS}"

exec "${PYTHON}" "${SCRIPT_DIR}/show_camera_topics.py" "$@"
