#!/usr/bin/env bash
# 在 Docker 容器内运行 viewer（ROS 在 Docker 内时必须用此脚本）
#
# 原因: Docker 内 ROS 使用 SHM 共享内存传输，图像数据无法传到宿主机进程。
#       topic 列表能发现，但收不到图像帧。
#
# 用法:
#   bash run_in_docker.sh
#   A2D_DOCKER_CONTAINER=my-container bash run_in_docker.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="${A2D_DOCKER_CONTAINER:-a2d-tele-release-2-1-0rc3-latest}"
REMOTE_DIR="/tmp/camera_topic_viewer"
A2D_SDK_HOME="${A2D_SDK_HOME:-${HOME}/a2d_sdk}"
FASTDDS_XML="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"

if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "错误: 容器 ${CONTAINER} 不存在" >&2
    echo "请设置: A2D_DOCKER_CONTAINER=你的容器名 bash run_in_docker.sh" >&2
    exit 1
fi

echo ">>> 容器: ${CONTAINER}"
echo ">>> 拷贝脚本到容器..."
docker exec "${CONTAINER}" mkdir -p "${REMOTE_DIR}"
docker cp "${SCRIPT_DIR}/show_camera_topics.py" "${CONTAINER}:${REMOTE_DIR}/"
if [[ -f "${FASTDDS_XML}" ]]; then
    docker exec "${CONTAINER}" mkdir -p /tmp/a2d_dds
    docker cp "${FASTDDS_XML}" "${CONTAINER}:/tmp/a2d_dds/fastdds_profiles_a2d.xml"
fi

DISPLAY="${DISPLAY:-:0}"
echo ">>> DISPLAY=${DISPLAY}"

docker exec \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/a2d_dds/fastdds_profiles_a2d.xml \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_DOMAIN_ID=0 \
    -e ROS_LOCALHOST_ONLY=1 \
    -it "${CONTAINER}" \
    bash -lc "
set -eo pipefail
source /opt/ros/humble/setup.bash
if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
    source /opt/psi/rt/a2d-tele/install/setup.bash
fi

# 清除 opencv 可能污染的 Qt 插件路径
unset QT_PLUGIN_PATH

# 安装 Python 依赖（容器内首次运行需要）
python3 -m pip install -q --user \
    'PyQt5==5.15.10' \
    'opencv-python-headless==4.10.0.84' \
    'numpy>=1.23.5,<2.0.0' \
    'pyqtgraph==0.13.7' \
    'PyOpenGL==3.1.7' 2>/dev/null || true
python3 -m pip uninstall -y opencv-python 2>/dev/null || true

echo '>>> 预检 topic...'
python3 - <<'PY'
import rclpy
rclpy.init()
from rclpy.node import Node
n = Node('preflight')
topics = [t for t in dict(n.get_topic_names_and_types()) if t.startswith('/camera')]
print(f'发现 {len(topics)} 个 /camera topic')
n.destroy_node()
rclpy.shutdown()
PY

exec python3 ${REMOTE_DIR}/show_camera_topics.py \"\$@\"
" -- "$@"
