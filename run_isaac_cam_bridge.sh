#!/usr/bin/env bash
# Isaac 共享帧 → ROS2 /camera/*_color 发布桥
#
# 本机通常没有 ROS2 Humble（在 a2d Docker 内），脚本会自动 docker exec 进容器启动。
# 共享目录必须放在宿主机与容器都能访问的路径（不要用 /tmp：容器常禁挂 /tmp）。
#
# 用法:
#   # 终端 A：ROS 桥（自动进 Docker）
#   bash run_isaac_cam_bridge.sh
#
#   # 终端 B：Isaac / RoboDojo（同一共享目录）
#   export ISAAC_CAM_BRIDGE_DIR=/share_data/projects/mahjong/share/personal/liyichao/eai/.cache/isaac_cam_bridge
#   # 然后启动评测
#
#   # 终端 C：viewer
#   bash run_in_docker.sh   # 勾选 /camera/head_color
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR_HOST="$(readlink -f "${SCRIPT_DIR}")"
CONTAINER="${A2D_DOCKER_CONTAINER:-a2d-tele-release-2-1-0rc3-latest}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
CONTAINER_USER="${HOST_UID}:${HOST_GID}"

# 默认放在 eai/.cache（share_data 路径，容器可读）；勿用 /tmp
DEFAULT_BRIDGE_DIR="${EAI_DIR_HOST}/.cache/isaac_cam_bridge"
export ISAAC_CAM_BRIDGE_DIR="${ISAAC_CAM_BRIDGE_DIR:-${DEFAULT_BRIDGE_DIR}}"
mkdir -p "${ISAAC_CAM_BRIDGE_DIR}"

PYTHON_IN_CONTAINER="${PYTHON_IN_CONTAINER:-python3.10}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

run_native() {
  local py="${PYTHON:-python3.10}"
  if ! command -v "${py}" >/dev/null 2>&1; then
    py="python3"
  fi
  # shellcheck disable=SC1091
  set +u
  source /opt/ros/humble/setup.bash
  set -e
  echo ">>> 模式: 宿主机原生 ROS"
  echo ">>> ISAAC_CAM_BRIDGE_DIR=${ISAAC_CAM_BRIDGE_DIR}"
  echo ">>> ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
  echo ">>> 发布: /camera/head_color /camera/left_wrist_color /camera/right_wrist_color"
  exec "${py}" "${EAI_DIR_HOST}/isaac_cam_ros_bridge.py" --dir "${ISAAC_CAM_BRIDGE_DIR}" "$@"
}

run_in_container() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "错误: 未找到 ROS2 Humble，且没有 docker 可用" >&2
    exit 1
  fi
  if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "错误: 容器 ${CONTAINER} 不存在。请先启动 a2d 容器，或设置 A2D_DOCKER_CONTAINER" >&2
    exit 1
  fi
  if ! docker exec -u "${CONTAINER_USER}" "${CONTAINER}" test -f "${EAI_DIR_HOST}/isaac_cam_ros_bridge.py"; then
    echo "错误: 容器内看不到 ${EAI_DIR_HOST}/isaac_cam_ros_bridge.py" >&2
    echo "  请确认 share_data / eai 已挂载进容器（与 run_in_docker.sh 相同）" >&2
    exit 1
  fi
  if ! docker exec -u "${CONTAINER_USER}" "${CONTAINER}" test -d "${ISAAC_CAM_BRIDGE_DIR}"; then
    docker exec -u "${CONTAINER_USER}" "${CONTAINER}" mkdir -p "${ISAAC_CAM_BRIDGE_DIR}" || true
  fi

  echo ">>> 模式: Docker 容器 ${CONTAINER}（宿主机无 ROS2）"
  echo ">>> ISAAC_CAM_BRIDGE_DIR=${ISAAC_CAM_BRIDGE_DIR}"
  echo ">>> ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"
  echo ">>> 发布: /camera/head_color /camera/left_wrist_color /camera/right_wrist_color"
  echo ">>> Isaac 侧请 export 同一 ISAAC_CAM_BRIDGE_DIR 后再开评测"

  # 与 viewer 同容器同 ROS_DOMAIN，才能互相看到 topic
  exec docker exec -i \
    -u "${CONTAINER_USER}" \
    -e "ISAAC_CAM_BRIDGE_DIR=${ISAAC_CAM_BRIDGE_DIR}" \
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" \
    -e "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}" \
    -e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}" \
    -e "HOME=/home/psibot" \
    -e "USER=psibot" \
    "${CONTAINER}" \
    bash -lc "
      set -e
      source /opt/ros/humble/setup.bash
      if [[ -f /home/psibot/workspace_liyichao/install/setup.bash ]]; then
        set +u
        source /home/psibot/workspace_liyichao/install/setup.bash
        set -e
      fi
      exec ${PYTHON_IN_CONTAINER} '${EAI_DIR_HOST}/isaac_cam_ros_bridge.py' \
        --dir '${ISAAC_CAM_BRIDGE_DIR}' --qos-best-effort $*
    "
}

if [[ -f /opt/ros/humble/setup.bash ]]; then
  run_native "$@"
else
  run_in_container "$@"
fi
