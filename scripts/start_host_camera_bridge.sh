#!/usr/bin/env bash
# 在 a2d Docker 容器内启动相机桥接，让宿主机 run_local.sh 能收到 /camera/* 图像。
#
# 原理: 容器内相机多为 FastDDS「仅 SHM」；本桥接用 SHM 收图，再经 UDP 发出。
# 宿主机请用:  bash run_local.sh（会自动对齐 Domain=0 + FastDDS）
#
# 用法:
#   bash scripts/start_host_camera_bridge.sh
#   bash scripts/start_host_camera_bridge.sh --stop
#   A2D_DOCKER_CONTAINER=xxx bash scripts/start_host_camera_bridge.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CONTAINER="${A2D_DOCKER_CONTAINER:-a2d-tele-release-2-1-0rc3-latest}"
BRIDGE_PY_HOST="${EAI_DIR}/camera_host_bridge.py"
DDS_XML_HOST="${EAI_DIR}/dds/fastdds_profiles_shm_udp.xml"
# 容器未必挂载 share_data：拷到容器内固定目录
REMOTE_DIR="/tmp/a2d_eai_camera_bridge"
BRIDGE_PY_CT="${REMOTE_DIR}/camera_host_bridge.py"
DDS_XML_CT="${REMOTE_DIR}/fastdds_profiles_shm_udp.xml"
LOG_FILE_CT="${REMOTE_DIR}/camera_host_bridge.log"
PID_FILE_HOST="${EAI_DIR}/.cache/camera_host_bridge.docker.pid"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

mkdir -p "${EAI_DIR}/.cache"

if [[ "${1:-}" == "--stop" || "${1:-}" == "stop" ]]; then
    if [[ -f "${PID_FILE_HOST}" ]]; then
        old="$(cat "${PID_FILE_HOST}" 2>/dev/null || true)"
        if [[ -n "${old}" ]]; then
            docker exec "${CONTAINER}" bash -lc "kill ${old} 2>/dev/null || true" || true
        fi
        rm -f "${PID_FILE_HOST}"
    fi
    docker exec "${CONTAINER}" bash -lc \
        "pkill -f 'camera_host_bridge.py' 2>/dev/null || true" || true
    echo ">>> 已停止 camera_host_bridge"
    exit 0
fi

if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "错误: 容器不存在: ${CONTAINER}" >&2
    exit 1
fi
if [[ ! -f "${BRIDGE_PY_HOST}" || ! -f "${DDS_XML_HOST}" ]]; then
    echo "错误: 缺少 ${BRIDGE_PY_HOST} 或 ${DDS_XML_HOST}" >&2
    exit 1
fi

echo ">>> 同步桥接脚本 → 容器 ${CONTAINER}:${REMOTE_DIR}"
docker exec "${CONTAINER}" bash -lc "mkdir -p '${REMOTE_DIR}' && chmod 777 '${REMOTE_DIR}'" || true
docker cp "${BRIDGE_PY_HOST}" "${CONTAINER}:${BRIDGE_PY_CT}"
docker cp "${DDS_XML_HOST}" "${CONTAINER}:${DDS_XML_CT}"

docker exec "${CONTAINER}" bash -lc "pkill -f 'camera_host_bridge.py' 2>/dev/null || true" || true

echo ">>> 启动 camera_host_bridge（SHM 收图 → UDP 给宿主机）"
# 与相机同用 root，避免 SHM/IPC 权限不一致
BRIDGE_PID="$(
docker exec \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e ROS_DOMAIN_ID=0 \
    -e ROS_LOCALHOST_ONLY=1 \
    -e "FASTRTPS_DEFAULT_PROFILES_FILE=${DDS_XML_CT}" \
    "${CONTAINER}" \
    bash -lc "
set -e
source /opt/ros/humble/setup.bash >/dev/null
if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
  source /opt/psi/rt/a2d-tele/install/setup.bash >/dev/null
fi
export FASTRTPS_DEFAULT_PROFILES_FILE='${DDS_XML_CT}'
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=1
cd '${REMOTE_DIR}'
: > '${LOG_FILE_CT}'
nohup python3 -u '${BRIDGE_PY_CT}' >> '${LOG_FILE_CT}' 2>&1 &
echo \$!
"
)"
BRIDGE_PID="$(echo "${BRIDGE_PID}" | tr -dc '0-9\n' | tail -n 1)"
if [[ -n "${BRIDGE_PID}" ]]; then
    echo "${BRIDGE_PID}" > "${PID_FILE_HOST}"
fi

sleep 1.5
if docker exec "${CONTAINER}" bash -lc "pgrep -af camera_host_bridge.py" >/dev/null 2>&1; then
    echo ">>> 桥接已启动 (pid=${BRIDGE_PID:-?})"
    echo ">>> 容器日志: docker exec ${CONTAINER} tail -f ${LOG_FILE_CT}"
    echo ">>> 宿主机启动界面:"
    echo "    cd ${EAI_DIR} && bash run_local.sh"
    echo ">>> 验证收图:"
    echo "    source /opt/ros/humble/setup.bash"
    echo "    export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_LOCALHOST_ONLY=1"
    echo "    export FASTRTPS_DEFAULT_PROFILES_FILE=${EAI_DIR}/dds/fastdds_profiles_host.xml"
    echo "    ros2 topic hz /camera/head_color"
else
    echo "错误: 桥接未起来" >&2
    docker exec "${CONTAINER}" bash -lc "tail -50 '${LOG_FILE_CT}' 2>/dev/null" || true
    exit 1
fi
