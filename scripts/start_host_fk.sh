#!/usr/bin/env bash
# 宿主机 FK：发布 /mink_fk/{left,right}_tcp_pose（无 Docker，替代 fk_node）
#
# 启动: bash scripts/start_host_fk.sh start
# 状态: bash scripts/start_host_fk.sh status
# 停止: bash scripts/start_host_fk.sh stop
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
FK_PY="${EAI_DIR}/host_fk_node.py"
CACHE_DIR="${EAI_DIR}/.cache"
LOG_FILE="${CACHE_DIR}/host_fk.log"
PID_FILE="${CACHE_DIR}/host_fk.pid"
DDS_XML="${EAI_DIR}/dds/fastdds_profiles_shm_udp.xml"
TELE_XML="${EAI_DIR}/dds/fastdds_profiles_tele.xml"
UDP_XML="${EAI_DIR}/dds/fastdds_profiles_udp_only.xml"
mkdir -p "${CACHE_DIR}"

_setup_ros() {
    unset COLCON_PREFIX_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH 2>/dev/null || true
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    # genie_msgs（WaistState）
    local genie_setup=""
    for cand in \
        "${EAI_GENIE_SETUP:-}" \
        /home/psibot/Projects/rcl/robot-control-library/install/setup.bash \
        /home/psibot/Projects/rcl/robot-control-library/install/local_setup.bash \
        /home/psibot/Projects/synrobot/install/setup.bash
    do
        if [[ -n "${cand}" && -f "${cand}" ]]; then
            genie_setup="${cand}"
            break
        fi
    done
    if [[ -n "${genie_setup}" ]]; then
        # shellcheck disable=SC1090
        source "${genie_setup}"
    fi
    set -u
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
    if [[ -f "${DDS_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_XML}"
    elif [[ -f "${UDP_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${UDP_XML}"
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
    pgrep -f '[h]ost_fk_node\.py' >/dev/null 2>&1
}

_other_fk_publisher() {
    # 避免 ros2 topic info（source+发现很慢）；只看官方 fk_node 进程
    if _running; then
        return 1
    fi
    pgrep -f '(^|/)fk_node( |$)|lib/wbc/fk_node' >/dev/null 2>&1
}

cmd_status() {
    if _running; then
        echo "host_fk: running"
        pgrep -af '[h]ost_fk_node\.py' | head -3
    else
        echo "host_fk: stopped"
    fi
    if command -v ros2 >/dev/null 2>&1; then
        _setup_ros
        echo "  /mink_fk/left_tcp_pose:"
        ros2 topic info /mink_fk/left_tcp_pose 2>/dev/null | sed 's/^/    /' || echo "    (n/a)"
    fi
}

_fk_python() {
    local candidate="${EAI_HOST_FK_PYTHON:-${EAI_DIR}/.cache/grasp_ik_env/bin/python}"
    if [[ -x "${candidate}" ]]; then
        echo "${candidate}"
    else
        echo /usr/bin/python3.10
    fi
}

cmd_start() {
    if [[ ! -f "${FK_PY}" ]]; then
        echo "错误: 缺少 ${FK_PY}" >&2
        exit 1
    fi
    if _running; then
        echo ">>> host_fk 已在运行"
        cmd_status
        return 0
    fi
    if _other_fk_publisher; then
        echo ">>> 已有 /mink_fk publisher，跳过 host_fk"
        cmd_status
        return 0
    fi
    _setup_ros
    local fk_python
    fk_python="$(_fk_python)"
    local fk_prefix fk_cmeel
    fk_prefix="$(dirname "$(dirname "${fk_python}")")"
    fk_cmeel="${fk_prefix}/lib/python3.10/site-packages/cmeel.prefix"
    if [[ -d "${fk_cmeel}" ]]; then
        export LD_LIBRARY_PATH="${fk_cmeel}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export PYTHONPATH="${fk_cmeel}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    if ! "${fk_python}" -c 'from genie_msgs.msg import WaistState' 2>/dev/null; then
        echo "错误: 无法 import genie_msgs.WaistState；请设置 EAI_GENIE_SETUP=.../install/setup.bash" >&2
        exit 1
    fi
    if ! "${fk_python}" -c 'import pinocchio' 2>/dev/null; then
        echo "错误: 需要 pinocchio（python3.10 -c 'import pinocchio'）" >&2
        exit 1
    fi
    echo ">>> 启动宿主机 FK（pinocchio → /mink_fk/*，无 Docker）"
    : >"${LOG_FILE}"
    setsid nohup "${fk_python}" "${FK_PY}" \
        >>"${LOG_FILE}" 2>&1 < /dev/null &
    echo $! >"${PID_FILE}"
    sleep 1.5
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
    pkill -f '[h]ost_fk_node\.py' 2>/dev/null || true
    echo ">>> host_fk stopped"
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
