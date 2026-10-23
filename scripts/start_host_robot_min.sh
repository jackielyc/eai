#!/usr/bin/env bash
# 宿主机最小机器人栈（无 Docker）：图像 + 手 + 手臂
#
# 启动: bash scripts/start_host_robot_min.sh start
# 停止: bash scripts/start_host_robot_min.sh stop
# 状态: bash scripts/start_host_robot_min.sh status
#
# 包含:
#   1) 相机 forwarder          → /camera/*（看图）
#   2) Ruiyan hand_node        → 灵巧手
#   3) host_arm_hand_bridge    → 使能 + UDP 中继（手/臂指令）
#   4) host_fk                 → /mink_fk/*_tcp_pose（臂移动需要 TCP）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_run() {
    local sh="$1"
    local act="$2"
    if [[ -f "${SCRIPT_DIR}/${sh}" ]]; then
        bash "${SCRIPT_DIR}/${sh}" "${act}" || true
    else
        echo "缺少 ${SCRIPT_DIR}/${sh}" >&2
    fi
}

cmd_start() {
    echo ">>> 启动最小栈：相机 + 手 + 手臂（无 Docker）"
    _run "start_cameras_minimal.sh" "start"
    _run "start_host_ruiyan.sh" "start"
    _run "start_host_arm_hand.sh" "start"
    _run "start_host_fk.sh" "start"
    echo ">>> 最小栈 start 完成"
}

cmd_stop() {
    echo ">>> 停止最小栈：相机 + 手 + 手臂（不关机器人本体）"
    # 先停指令桥，再停 FK / 手 / 相机
    _run "start_host_arm_hand.sh" "stop"
    _run "start_host_fk.sh" "stop"
    _run "start_host_ruiyan.sh" "stop"
    _run "start_cameras_minimal.sh" "stop"
    # TCP bridge 由 UI 管理；这里顺带清残留
    pkill -f '/usr/bin/python3\.10 .*robot_tcp_bridge\.py' 2>/dev/null || true
    echo ">>> 最小栈 stop 完成"
}

cmd_status() {
    echo "=== host robot min ==="
    _run "start_cameras_minimal.sh" "status"
    echo
    _run "start_host_ruiyan.sh" "status"
    echo
    _run "start_host_arm_hand.sh" "status"
    echo
    _run "start_host_fk.sh" "status"
    echo
    if pgrep -f '/usr/bin/python3\.10 .*robot_tcp_bridge\.py' >/dev/null 2>&1; then
        echo "robot_tcp_bridge: running"
        pgrep -af '/usr/bin/python3\.10 .*robot_tcp_bridge\.py' | head -3
    else
        echo "robot_tcp_bridge: stopped"
    fi
}

case "${1:-status}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    status|"") cmd_status ;;
    -h|--help|help)
        sed -n '2,14p' "$0" | sed 's/^# \?//'
        ;;
    *)
        echo "未知命令: $1" >&2
        exit 1
        ;;
esac
