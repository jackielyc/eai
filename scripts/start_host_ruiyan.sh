#!/usr/bin/env bash
# 宿主机 Ruiyan 灵巧手驱动（永远不用 Docker）
#
# 依赖: eai/.cache/host_ruiyan_ws 已 colcon 编译 hand
# 串口: /dev/ry_hand_left、/dev/ry_hand_right（或通过 RY_HAND_*_PORT 覆盖）
#
# 启动: bash scripts/start_host_ruiyan.sh start
# 状态: bash scripts/start_host_ruiyan.sh status
# 停止: bash scripts/start_host_ruiyan.sh stop
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
WS="${EAI_DIR}/.cache/host_ruiyan_ws"
CACHE_DIR="${EAI_DIR}/.cache"
LOG_FILE="${CACHE_DIR}/host_ruiyan.log"
PID_FILE="${CACHE_DIR}/host_ruiyan.pid"
DDS_XML="${EAI_DIR}/dds/fastdds_profiles_shm_udp.xml"
TELE_XML="${EAI_DIR}/dds/fastdds_profiles_tele.xml"
CONFIG_NAME="${RY_HAND_CONFIG:-ry_host_serial}"
mkdir -p "${CACHE_DIR}"

_setup_ros() {
    unset COLCON_PREFIX_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH 2>/dev/null || true
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    # shellcheck disable=SC1091
    source "${WS}/install/setup.bash"
    set -u
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
    # conda glog/protobuf at runtime
    export LD_LIBRARY_PATH="/opt/miniconda3/lib:${LD_LIBRARY_PATH:-}"
    if [[ -f "${DDS_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_XML}"
    elif [[ -f "${TELE_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${TELE_XML}"
    fi
}

_ensure_serial_symlinks() {
    # Optional override: RY_HAND_LEFT_PORT=/dev/ttyUSB0 RY_HAND_RIGHT_PORT=/dev/ttyUSB1
    local left="${RY_HAND_LEFT_PORT:-}"
    local right="${RY_HAND_RIGHT_PORT:-}"
    if [[ -n "${left}" && -e "${left}" && ! -e /dev/ry_hand_left ]]; then
        ln -sfn "${left}" /dev/ry_hand_left 2>/dev/null || \
            sudo ln -sfn "${left}" /dev/ry_hand_left 2>/dev/null || true
    fi
    if [[ -n "${right}" && -e "${right}" && ! -e /dev/ry_hand_right ]]; then
        ln -sfn "${right}" /dev/ry_hand_right 2>/dev/null || \
            sudo ln -sfn "${right}" /dev/ry_hand_right 2>/dev/null || true
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
    pgrep -f '/hand/hand_node|lib/hand/hand_node' >/dev/null 2>&1
}

cmd_status() {
    if _running; then
        echo "host_ruiyan (hand_node): running"
        pgrep -af 'hand_node' | head -5
    else
        echo "host_ruiyan (hand_node): stopped"
    fi
    for d in /dev/ry_hand_left /dev/ry_hand_right; do
        if [[ -e "${d}" ]]; then
            echo "  ${d}: ok -> $(readlink -f "${d}" 2>/dev/null || echo present)"
        else
            echo "  ${d}: MISSING"
        fi
    done
    if command -v ros2 >/dev/null 2>&1; then
        _setup_ros
        echo "  set_angles subscribers:"
        ros2 topic info /ry_hand/left/set_angles 2>/dev/null | sed 's/^/    /' || echo "    (left: n/a)"
        ros2 topic info /ry_hand/right/set_angles 2>/dev/null | sed 's/^/    /' || echo "    (right: n/a)"
    fi
}

cmd_start() {
    if [[ ! -x "${WS}/install/hand/lib/hand/hand_node" ]]; then
        echo "错误: 未找到 hand_node，请先编译 ${WS}" >&2
        echo "  bash --noprofile --norc -c 'source /opt/ros/humble/setup.bash && cd ${WS} && colcon build --packages-up-to hand'" >&2
        exit 1
    fi
    if _running; then
        echo ">>> host_ruiyan 已在运行"
        cmd_status
        return 0
    fi
    _ensure_serial_symlinks
    if [[ ! -e /dev/ry_hand_left || ! -e /dev/ry_hand_right ]]; then
        echo "警告: 串口设备缺失（期望 /dev/ry_hand_{left,right}）。" >&2
        echo "  可设置 RY_HAND_LEFT_PORT / RY_HAND_RIGHT_PORT 后重试。" >&2
        ls -la /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
    fi
    _setup_ros
    local bin="${WS}/install/hand/lib/hand/hand_node"
    if [[ ! -x "${bin}" ]]; then
        echo "错误: 找不到可执行文件 ${bin}" >&2
        exit 1
    fi
    echo ">>> 启动宿主机 Ruiyan hand_node（config=${CONFIG_NAME}，无 Docker）"
    # 直接二进制 + 新会话，避免 ros2 CLI 包装进程退出/被误杀带走驱动
    : >"${LOG_FILE}"
    setsid nohup "${bin}" --ros-args -p "config_file:=${CONFIG_NAME}" \
        >>"${LOG_FILE}" 2>&1 < /dev/null &
    echo $! >"${PID_FILE}"
    sleep 2
    if _running; then
        echo ">>> 已启动 pid=$(cat "${PID_FILE}") log=${LOG_FILE}"
        cmd_status
    else
        echo "启动失败，见 ${LOG_FILE}" >&2
        tail -n 60 "${LOG_FILE}" >&2 || true
        exit 1
    fi
}

cmd_stop() {
    if [[ -f "${PID_FILE}" ]]; then
        kill "$(cat "${PID_FILE}")" 2>/dev/null || true
        rm -f "${PID_FILE}"
    fi
    pkill -f 'lib/hand/hand_node' 2>/dev/null || true
    pkill -f 'hand hand_node' 2>/dev/null || true
    echo ">>> host_ruiyan stopped"
}

case "${1:-status}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    status|"") cmd_status ;;
    -h|--help|help)
        sed -n '2,12p' "$0" | sed 's/^# \?//'
        ;;
    *)
        echo "未知命令: $1" >&2
        exit 1
        ;;
esac
