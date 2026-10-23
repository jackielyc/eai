#!/usr/bin/env bash
# 宿主机手/臂使能桥（永远不用 Docker）
#
# 启动: bash scripts/start_host_arm_hand.sh start
# 状态: bash scripts/start_host_arm_hand.sh status
# 停止: bash scripts/start_host_arm_hand.sh stop
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
BRIDGE_PY="${SCRIPT_DIR}/host_arm_hand_bridge.py"
CACHE_DIR="${EAI_DIR}/.cache"
LOG_FILE="${CACHE_DIR}/host_arm_hand_bridge.log"
PID_FILE="${CACHE_DIR}/host_arm_hand_bridge.pid"
DDS_XML="${EAI_DIR}/dds/fastdds_profiles_shm_udp.xml"
TELE_XML="${EAI_DIR}/dds/fastdds_profiles_tele.xml"
RELAY_PORT="${HAND_CMD_RELAY_PORT:-19283}"
mkdir -p "${CACHE_DIR}"

_setup_ros() {
    unset COLCON_PREFIX_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH 2>/dev/null || true
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -u
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=0
    export ROS_LOCALHOST_ONLY=1
    if [[ -f "${DDS_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_XML}"
    elif [[ -f "${TELE_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${TELE_XML}"
    fi
}

_running() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    fi
    pgrep -f 'host_arm_hand_bridge\.py' >/dev/null 2>&1
}

cmd_status() {
    if _running; then
        echo "host_arm_hand_bridge: running"
        pgrep -af 'host_arm_hand_bridge\.py' | head -3
    else
        echo "host_arm_hand_bridge: stopped"
    fi
    if ss -ulnp 2>/dev/null | grep -qE ":${RELAY_PORT}\\b"; then
        echo "udp relay :${RELAY_PORT}: listening"
    else
        echo "udp relay :${RELAY_PORT}: (no listener)"
    fi
}

cmd_start() {
    if [[ ! -f "${BRIDGE_PY}" ]]; then
        echo "错误: 缺少 ${BRIDGE_PY}" >&2
        exit 1
    fi
    if _running; then
        echo ">>> host_arm_hand_bridge 已在运行"
        cmd_status
        return 0
    fi
    _setup_ros
    echo ">>> 启动宿主机手/臂使能桥（无 Docker）"
    nohup /usr/bin/python3.10 "${BRIDGE_PY}" --port "${RELAY_PORT}" \
        >>"${LOG_FILE}" 2>&1 &
    echo $! >"${PID_FILE}"
    sleep 1
    if _running; then
        echo ">>> 已启动 pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
    else
        echo "启动失败，见 ${LOG_FILE}" >&2
        tail -n 40 "${LOG_FILE}" >&2 || true
        exit 1
    fi
}

cmd_stop() {
    if [[ -f "${PID_FILE}" ]]; then
        kill "$(cat "${PID_FILE}")" 2>/dev/null || true
        rm -f "${PID_FILE}"
    fi
    pkill -f 'host_arm_hand_bridge\.py' 2>/dev/null || true
    echo ">>> host_arm_hand_bridge stopped"
}

case "${1:-status}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    status|"") cmd_status ;;
    -h|--help|help)
        sed -n '2,8p' "$0" | sed 's/^# \?//'
        ;;
    *)
        echo "未知命令: $1" >&2
        exit 1
        ;;
esac
