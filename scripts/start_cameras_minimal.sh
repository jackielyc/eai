#!/usr/bin/env bash
# 读取机器人头/腕相机（宿主机 a2d_sdk forwarder，不使用本地 Docker）
#
# 原理: 机器人(10.42.0.101) 上 Cosine 相机 → 本机 a2d_sdk forwarder 桥成 ROS
#       /camera/head_color、/camera/hand_left_color、/camera/hand_right_color
#
# 前置:
#   - 网卡在 10.42.0.*（本机已常见）
#   - ~/a2d_sdk 或 A2D_SDK_HOME 指向带 forwarder 的 SDK
#   - 机器人侧相机已在跑（否则需 robot-service -s 切模式，见下方）
#
# 用法:
#   bash scripts/start_cameras_minimal.sh status
#   bash scripts/start_cameras_minimal.sh start          # 只起 forwarder
#   bash scripts/start_cameras_minimal.sh robot-service  # 可选：切模式+相机（慢）
#   bash scripts/start_cameras_minimal.sh hz
#   bash scripts/start_cameras_minimal.sh stop
#
# 然后: ./scripts/launch/run_local.sh →「连接相机 Topic」勾选 head/hand_*_color
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
A2D_SDK_HOME="${A2D_SDK_HOME:-${HOME}/a2d_sdk}"
# 兼容旧布局 ~/a2d_sdk/a2d_sdk
if [[ -d "${A2D_SDK_HOME}/a2d_sdk/forwarder" && ! -d "${A2D_SDK_HOME}/forwarder" ]]; then
    A2D_SDK_HOME="${A2D_SDK_HOME}/a2d_sdk"
fi
WRAPPER="${A2D_SDK_HOME}/scripts/ros_env_wrapper.sh"
FORWARDER_BIN="${A2D_SDK_HOME}/forwarder/app/bin/forwarder"
ROBOT_SERVICE_BIN="${ROBOT_SERVICE_BIN:-}"
for c in \
    "${HOME}/Downloads/a2d/bin/robot-service" \
    /usr/local/bin/robot-service \
    "${A2D_SDK_HOME}/../bin/robot-service"; do
    if [[ -z "${ROBOT_SERVICE_BIN}" && -x "${c}" ]]; then
        ROBOT_SERVICE_BIN="${c}"
    fi
done
A2D_XML="${EAI_DIR}/dds/fastdds_profiles_a2d.xml"
CACHE_DIR="${EAI_DIR}/.cache"
LOG_FILE="${CACHE_DIR}/host_forwarder.log"
mkdir -p "${CACHE_DIR}"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \?//'
}

_check_net() {
    local_ip="$(ip -o -4 addr list 2>/dev/null | awk '/10\.42\.0\./{print $4}' | cut -d/ -f1 | head -1)"
    if [[ -z "${local_ip}" ]]; then
        echo "错误: 本机没有 10.42.0.* 地址，无法连机器人" >&2
        exit 1
    fi
    echo ">>> LOCATOR_IP=${local_ip}  robot=10.42.0.101"
    if ! ping -c 1 -W 1 10.42.0.101 >/dev/null 2>&1; then
        echo "警告: ping 10.42.0.101 失败，forwarder 可能收不到相机" >&2
    fi
}

_setup_ros() {
    unset COLCON_PREFIX_PATH 2>/dev/null || true
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -u
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID=0
    export ROS_LOCALHOST_ONLY=1
    # 宿主机进程统一 shm+UDP，避免仅 SHM 时端口锁导致发现失败
    SHM_UDP_XML="${EAI_DIR}/dds/fastdds_profiles_shm_udp.xml"
    TELE_XML="${EAI_DIR}/dds/fastdds_profiles_tele.xml"
    if [[ -f "${SHM_UDP_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${SHM_UDP_XML}"
    elif [[ -f "${TELE_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${TELE_XML}"
    elif [[ -f "${A2D_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${A2D_XML}"
    fi
}

cmd_status() {
    _check_net
    echo "A2D_SDK_HOME=${A2D_SDK_HOME}"
    if [[ -x "${FORWARDER_BIN}" ]]; then
        echo "forwarder bin: ok"
    else
        echo "forwarder bin: MISSING (${FORWARDER_BIN})"
    fi
    if pgrep -f "${FORWARDER_BIN}" >/dev/null 2>&1; then
        echo "forwarder: running"
        pgrep -af "${FORWARDER_BIN}" | head -3
    else
        echo "forwarder: stopped"
    fi
    if pgrep -af '[r]obot-service' >/dev/null 2>&1; then
        echo "robot-service: running"
    else
        echo "robot-service: stopped"
    fi
    _setup_ros
    echo "camera topics:"
    ros2 topic list 2>/dev/null | grep -E '^/camera/(head|hand_).*(color|depth)$' | sed 's/^/  /' \
        || echo "  (无 — 先 start，并确认机器人相机已开)"
}

cmd_start() {
    _check_net
    if [[ ! -x "${FORWARDER_BIN}" ]]; then
        echo "错误: 找不到 forwarder: ${FORWARDER_BIN}" >&2
        echo "  设置 A2D_SDK_HOME=... 指向含 forwarder 的 SDK" >&2
        exit 1
    fi
    if [[ ! -x "${WRAPPER}" ]]; then
        echo "错误: 找不到 ${WRAPPER}" >&2
        exit 1
    fi
    if pgrep -f "${FORWARDER_BIN}" >/dev/null 2>&1; then
        echo ">>> forwarder 已在运行"
    else
        echo ">>> 启动宿主机 forwarder（桥接机器人相机 → ROS /camera/*）"
        # ros_env_wrapper 会 source env.sh（设 LOCATOR_IP / AORTA）
        nohup bash "${WRAPPER}" start >>"${LOG_FILE}" 2>&1 || true
        sleep 2
        if pgrep -f "${FORWARDER_BIN}" >/dev/null 2>&1; then
            echo ">>> forwarder 已启动"
        else
            echo "forwarder 启动失败，见 ${LOG_FILE}" >&2
            # 直接兜底启动
            set +u
            # shellcheck disable=SC1091
            source "${A2D_SDK_HOME}/env.sh"
            set -u
            export PATH="${A2D_SDK_HOME}/forwarder/app/bin:${PATH:-}"
            export LD_LIBRARY_PATH="${A2D_SDK_HOME}/forwarder/app/lib:${LD_LIBRARY_PATH:-}"
            nohup "${FORWARDER_BIN}" >>"${LOG_FILE}" 2>&1 &
            sleep 2
            pgrep -af "${FORWARDER_BIN}" | head -3 || {
                echo "仍失败" >&2
                tail -n 30 "${LOG_FILE}" >&2 || true
                exit 1
            }
        fi
    fi
    echo ">>> 验证: $0 hz"
    echo "    UI: ./scripts/launch/run_local.sh → 连接相机 Topic → 勾选 head/hand_*_color"
}

cmd_robot_service() {
    _check_net
    if [[ -z "${ROBOT_SERVICE_BIN}" || ! -x "${ROBOT_SERVICE_BIN}" ]]; then
        echo "错误: 未找到 robot-service 可执行文件" >&2
        echo "  常见路径: ~/Downloads/a2d/bin/robot-service" >&2
        exit 1
    fi
    conf="${A2D_SDK_HOME}/conf/copilot.pbtxt"
    if [[ ! -f "${conf}" ]]; then
        echo "错误: 缺少 ${conf}" >&2
        exit 1
    fi
    if pgrep -af '[r]obot-service' >/dev/null 2>&1; then
        echo ">>> robot-service 已在运行"
        exit 0
    fi
    echo ">>> 启动 robot-service -s（切模式+相机，可能数分钟）"
    set +u
    # shellcheck disable=SC1091
    source "${A2D_SDK_HOME}/env.sh"
    set -u
    if [[ -f "${A2D_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${A2D_XML}"
    fi
    nohup "${ROBOT_SERVICE_BIN}" -s -c "${conf}" >>"${CACHE_DIR}/host_robot_service.log" 2>&1 &
    echo $! >"${CACHE_DIR}/host_robot_service.pid"
    echo ">>> pid=$(cat "${CACHE_DIR}/host_robot_service.pid") log=${CACHE_DIR}/host_robot_service.log"
    echo "    完成后请再: $0 start && $0 hz"
}

cmd_hz() {
    _setup_ros
    echo ">>> ros2 topic hz（Ctrl+C 停）"
    for t in /camera/head_color /camera/hand_left_color /camera/hand_right_color; do
        echo "---- ${t} ----"
        timeout 3 ros2 topic hz "${t}" 2>&1 | tail -n 5 || echo "(无数据)"
    done
}

cmd_stop() {
    echo ">>> 停止宿主机 forwarder（不关机器人）"
    if [[ -x "${WRAPPER}" ]]; then
        bash "${WRAPPER}" stop 2>/dev/null || true
    fi
    pkill -f "${FORWARDER_BIN}" 2>/dev/null || true
    if [[ -f "${CACHE_DIR}/host_robot_service.pid" ]]; then
        kill "$(cat "${CACHE_DIR}/host_robot_service.pid")" 2>/dev/null || true
        rm -f "${CACHE_DIR}/host_robot_service.pid"
    fi
    # 不默认杀所有 robot-service，避免误伤；需要时:
    #   pkill -f robot-service
    echo ">>> done（若仍要停 robot-service: pkill -f robot-service）"
}

ACTION="${1:-}"
case "${ACTION}" in
    status|"")
        [[ -z "${ACTION}" ]] && usage && echo
        cmd_status
        ;;
    start|forwarder)
        cmd_start
        ;;
    robot-service|robot_service|rs)
        cmd_robot_service
        ;;
    hz)
        cmd_hz
        ;;
    stop)
        cmd_stop
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "未知命令: ${ACTION}" >&2
        usage
        exit 1
        ;;
esac
