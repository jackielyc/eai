#!/usr/bin/env bash
# 宿主机 RViz：robot_description + /joint_states TF + rviz2（无 Docker）
#
# 启动: bash scripts/start_host_rviz.sh start
# 只开可视化节点: bash scripts/start_host_rviz.sh viz
# 停止: bash scripts/start_host_rviz.sh stop
# 状态: bash scripts/start_host_rviz.sh status
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CACHE_DIR="${EAI_DIR}/.cache"
LOG_FILE="${CACHE_DIR}/host_rviz.log"
RSP_PID_FILE="${CACHE_DIR}/host_rviz_rsp.pid"
STATIC_PID_FILE="${CACHE_DIR}/host_rviz_static_tf.pid"
RVIZ_PID_FILE="${CACHE_DIR}/host_rviz_gui.pid"
DEPTH_PID_FILE="${CACHE_DIR}/host_rviz_depth.pid"
DEPTH_NODE="${EAI_DIR}/compat/depth_cloud_node.py"
URDF_OUT="${CACHE_DIR}/A2D_NoHand_Flattened.rviz.urdf"
RSP_PARAMS="${CACHE_DIR}/host_rviz_rsp_params.yaml"
FK_SCRIPT="${SCRIPT_DIR}/start_host_fk.sh"
DDS_XML="${EAI_DIR}/dds/fastdds_profiles_shm_udp.xml"
TELE_XML="${EAI_DIR}/dds/fastdds_profiles_tele.xml"
UDP_XML="${EAI_DIR}/dds/fastdds_profiles_udp_only.xml"
mkdir -p "${CACHE_DIR}"

_resolve_urdf() {
    if [[ -n "${EAI_HOST_FK_URDF:-}" && -f "${EAI_HOST_FK_URDF}" ]]; then
        echo "${EAI_HOST_FK_URDF}"
        return
    fi
    local cand
    for cand in \
        "${EAI_DIR}/../a2d-tele/modules/control/wbc/assets/A2D_NoHand/A2D_NoHand_Flattened.urdf" \
        "${EAI_DIR}/a2d-tele/modules/control/wbc/assets/A2D_NoHand/A2D_NoHand_Flattened.urdf" \
        "/share_data/projects/mahjong/share/personal/liyichao/a2d-tele/modules/control/wbc/assets/A2D_NoHand/A2D_NoHand_Flattened.urdf"
    do
        if [[ -f "${cand}" ]]; then
            echo "${cand}"
            return
        fi
    done
    return 1
}

_resolve_rviz_config() {
    local cand patched
    patched="${CACHE_DIR}/fk_visualization.host.rviz"
    for cand in \
        "${EAI_DIR}/../a2d-tele/modules/control/wbc/config/fk_visualization.rviz" \
        "${EAI_DIR}/a2d-tele/modules/control/wbc/config/fk_visualization.rviz" \
        "/share_data/projects/mahjong/share/personal/liyichao/a2d-tele/modules/control/wbc/config/fk_visualization.rviz"
    do
        if [[ -f "${cand}" ]]; then
            # robot_state_publisher 的 /robot_description 是 Transient Local。
            # 原配置写成 Volatile，RViz 接不上模型，画面里就没有机器人。
            /usr/bin/python3.10 - "${cand}" "${patched}" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
text = text.replace(
    "      Description Topic:\n        Depth: 5\n        Durability Policy: Volatile\n",
    "      Description Topic:\n        Depth: 5\n        Durability Policy: Transient Local\n",
)
clouds = """
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: rgb
      Class: rviz_default_plugins/PointCloud2
      Color: 80; 220; 220
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: 头部深度
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.01
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /eai/depth_cloud/head
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: rgb
      Class: rviz_default_plugins/PointCloud2
      Color: 70; 140; 230
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: 左腕深度
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.01
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /eai/depth_cloud/hand_left
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: rgb
      Class: rviz_default_plugins/PointCloud2
      Color: 230; 140; 60
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: 右腕深度
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.01
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /eai/depth_cloud/hand_right
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: 相机图像
      Normalize Range: false
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /eai/color_mosaic
      Value: true
"""
needle = "\n  Enabled: true\n  Global Options:"
if needle not in text:
    raise SystemExit("rviz config has no Displays end marker")
if "/eai/depth_cloud/head" not in text:
    text = text.replace(needle, "\n" + clouds + needle, 1)
# 带三路图像窗口的旧布局会让 Ogre 创建失败，RViz 重试约 100 次后退出。
lines = [ln for ln in text.splitlines(keepends=True) if not ln.startswith("  QMainWindow State:")]
open(dst, "w", encoding="utf-8").write("".join(lines))
PY
            echo "${patched}"
            return
        fi
    done
    echo ""
}

_setup_ros() {
    unset COLCON_PREFIX_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH 2>/dev/null || true
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
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
    # 与 UI / forwarder 一致：本机可见
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
    if [[ -f "${DDS_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${DDS_XML}"
    elif [[ -f "${TELE_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${TELE_XML}"
    elif [[ -f "${UDP_XML}" ]]; then
        export FASTRTPS_DEFAULT_PROFILES_FILE="${UDP_XML}"
    fi
    # conda 的 libcurl 会盖住 RViz 的 resource_retriever，网格加载失败时机器人是空的
    local cleaned="" part
    IFS=':' read -ra _ld_parts <<< "${LD_LIBRARY_PATH:-}"
    for part in "${_ld_parts[@]}"; do
        [[ -z "${part}" ]] && continue
        [[ "${part}" == *miniconda* || "${part}" == *anaconda* || "${part}" == *conda* ]] && continue
        cleaned="${cleaned:+${cleaned}:}${part}"
    done
    export LD_LIBRARY_PATH="/opt/ros/humble/lib:/usr/lib/x86_64-linux-gnu${cleaned:+:${cleaned}}"
}

_pid_alive() {
    local file="$1"
    local needle="$2"
    if [[ ! -f "${file}" ]]; then
        return 1
    fi
    local pid
    pid="$(cat "${file}" 2>/dev/null || true)"
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        return 1
    fi
    if [[ -n "${needle}" ]]; then
        tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null | grep -q "${needle}" || return 1
    fi
    return 0
}

_prepare_urdf() {
    local src
    src="$(_resolve_urdf)" || {
        echo "错误: 找不到 A2D_NoHand_Flattened.urdf" >&2
        exit 1
    }
    local mesh_root
    mesh_root="$(cd "$(dirname "${src}")" && pwd -P)"
    # package://wbc/... 在宿主机无 ament 包；改成绝对 file:// 才能加载网格
    /usr/bin/python3.10 - "${src}" "${URDF_OUT}" "${mesh_root}" <<'PY'
import sys
src, out, mesh_root = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src, encoding="utf-8").read()
# package://wbc/assets/A2D_NoHand/meshes/foo.obj -> file:///.../A2D_NoHand/meshes/foo.obj
old = "package://wbc/assets/A2D_NoHand/"
new = "file://" + mesh_root.rstrip("/") + "/"
if old not in text:
    # 兜底：任意 package://wbc/assets/ 前缀
    old = "package://wbc/assets/"
    parent = mesh_root.rstrip("/").rsplit("/", 1)[0]
    new = "file://" + parent + "/"
text = text.replace(old, new)
open(out, "w", encoding="utf-8").write(text)
print(out)
PY
}

_start_rsp() {
    if _pid_alive "${RSP_PID_FILE}" robot_state_publisher; then
        echo ">>> robot_state_publisher 已在运行"
        return 0
    fi
    if ! command -v ros2 >/dev/null 2>&1; then
        echo "错误: 未找到 ros2" >&2
        exit 1
    fi
    if [[ ! -x /opt/ros/humble/lib/robot_state_publisher/robot_state_publisher ]]; then
        echo "错误: 未安装 robot_state_publisher（apt install ros-humble-robot-state-publisher）" >&2
        exit 1
    fi
    _prepare_urdf
    /usr/bin/python3.10 - "${URDF_OUT}" "${RSP_PARAMS}" <<'PY'
import json, sys
urdf = open(sys.argv[1], encoding="utf-8").read()
# 手工拼最小 YAML：robot_description 用 JSON 引号转义
desc = json.dumps(urdf, ensure_ascii=False)
open(sys.argv[2], "w", encoding="utf-8").write(
    "eai_host_robot_state_publisher:\n"
    "  ros__parameters:\n"
    f"    robot_description: {desc}\n"
    "    publish_frequency: 50.0\n"
    "    ignore_timestamp: true\n"
)
print(sys.argv[2])
PY
    echo ">>> 启动 robot_state_publisher（/joint_states → TF）"
    # 直接跑 lib 里的二进制：ros2 run 包装进程会立刻退出，PID 不可靠
    setsid nohup /opt/ros/humble/lib/robot_state_publisher/robot_state_publisher \
        --ros-args \
        -r __node:=eai_host_robot_state_publisher \
        --params-file "${RSP_PARAMS}" \
        >>"${LOG_FILE}" 2>&1 < /dev/null &
    echo $! >"${RSP_PID_FILE}"
    sleep 1.2
    if _pid_alive "${RSP_PID_FILE}" robot_state_publisher; then
        echo ">>> rsp pid=$(cat "${RSP_PID_FILE}")"
    else
        echo "robot_state_publisher 启动失败，见 ${LOG_FILE}" >&2
        tail -n 40 "${LOG_FILE}" >&2 || true
        exit 1
    fi
}

_start_static_tf() {
    if _pid_alive "${STATIC_PID_FILE}" static_transform_publisher; then
        echo ">>> map→A2D 静态 TF 已在运行"
        return 0
    fi
    if [[ ! -x /opt/ros/humble/lib/tf2_ros/static_transform_publisher ]]; then
        echo "警告: 无 static_transform_publisher，RViz Fixed Frame 请改成 A2D" >&2
        return 0
    fi
    echo ">>> 启动静态 TF map → A2D"
    setsid nohup /opt/ros/humble/lib/tf2_ros/static_transform_publisher \
        --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 \
        --frame-id map --child-frame-id A2D \
        --ros-args -r __node:=eai_host_map_to_a2d \
        >>"${LOG_FILE}" 2>&1 < /dev/null &
    echo $! >"${STATIC_PID_FILE}"
    sleep 0.5
}

_start_rviz() {
    if _pid_alive "${RVIZ_PID_FILE}" rviz2; then
        echo ">>> RViz 已在运行 pid=$(cat "${RVIZ_PID_FILE}")"
        return 0
    fi
    if ! command -v ros2 >/dev/null 2>&1; then
        echo "错误: 未找到 ros2 / rviz2" >&2
        exit 1
    fi
    local cfg
    cfg="$(_resolve_rviz_config)"
    local rviz_bin=""
    for cand in /opt/ros/humble/lib/rviz2/rviz2 "$(command -v rviz2 2>/dev/null || true)"; do
        if [[ -n "${cand}" && -x "${cand}" ]]; then
            rviz_bin="${cand}"
            break
        fi
    done
    if [[ -z "${rviz_bin}" ]]; then
        echo "错误: 未找到 rviz2 可执行文件" >&2
        exit 1
    fi
    echo ">>> 启动 rviz2 (${rviz_bin})"
    if [[ -n "${cfg}" ]]; then
        setsid nohup "${rviz_bin}" -d "${cfg}" >>"${LOG_FILE}" 2>&1 < /dev/null &
    else
        setsid nohup "${rviz_bin}" >>"${LOG_FILE}" 2>&1 < /dev/null &
    fi
    echo $! >"${RVIZ_PID_FILE}"
    sleep 0.8
    if _pid_alive "${RVIZ_PID_FILE}" rviz2; then
        echo ">>> rviz pid=$(cat "${RVIZ_PID_FILE}") config=${cfg:-none}"
    else
        echo "rviz2 启动失败，见 ${LOG_FILE}" >&2
        tail -n 40 "${LOG_FILE}" >&2 || true
        exit 1
    fi
}

cmd_viz() {
    _setup_ros
    : >>"${LOG_FILE}"
    if [[ -x "${FK_SCRIPT}" ]]; then
        # 与 UI 同 Domain 可见：覆盖 start_host_fk 默认 LOCALHOST_ONLY=1
        export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
        # 需要带 /joint_states 的新版 host_fk；旧进程则重启
        if pgrep -f '/(python|python3\.10) .*host_fk_node\.py' >/dev/null 2>&1; then
            if ! timeout 4 ros2 topic info /joint_states 2>/dev/null | grep -q "Publisher count: [1-9]"; then
                echo ">>> 重启 host_fk 以发布 /joint_states"
                bash "${FK_SCRIPT}" stop || true
                sleep 0.5
            fi
        fi
        bash "${FK_SCRIPT}" start || true
        sleep 1
    fi
    _start_rsp
    _start_static_tf
    _start_depth
}

_start_depth() {
    if _pid_alive "${DEPTH_PID_FILE}" compat/depth_cloud_node.py; then
        echo ">>> 头/腕深度点云已在运行"
        return 0
    fi
    if [[ ! -f "${DEPTH_NODE}" ]]; then
        echo "警告: 缺少 ${DEPTH_NODE}，RViz 里没有深度点云" >&2
        return 0
    fi
    echo ">>> 启动头/腕深度点云"
    setsid nohup /usr/bin/python3.10 "${DEPTH_NODE}" >>"${LOG_FILE}" 2>&1 < /dev/null &
    echo $! >"${DEPTH_PID_FILE}"
    sleep 0.6
    if _pid_alive "${DEPTH_PID_FILE}" compat/depth_cloud_node.py; then
        echo ">>> depth pid=$(cat "${DEPTH_PID_FILE}")"
    else
        echo "深度点云启动失败，见 ${LOG_FILE}" >&2
        tail -n 30 "${LOG_FILE}" >&2 || true
    fi
}

cmd_start() {
    cmd_viz
    _start_rviz
    echo ">>> 完成。RViz Fixed Frame=map；机器人来自 /joint_states + robot_description"
}

cmd_stop() {
    for f in "${RVIZ_PID_FILE}" "${RSP_PID_FILE}" "${STATIC_PID_FILE}" "${DEPTH_PID_FILE}"; do
        if [[ -f "${f}" ]]; then
            kill "$(cat "${f}")" 2>/dev/null || true
            rm -f "${f}"
        fi
    done
    pkill -f 'robot_state_publisher .*eai_host_robot_state_publisher|__node:=eai_host_robot_state_publisher' 2>/dev/null || true
    pkill -f 'eai_host_map_to_a2d' 2>/dev/null || true
    pkill -f 'compat/depth_cloud_node.py' 2>/dev/null || true
    echo ">>> host_rviz stopped（未停 host_fk）"
}

cmd_status() {
    echo "depth: $(_pid_alive "${DEPTH_PID_FILE}" compat/depth_cloud_node.py && echo running || echo stopped)"
    echo "rsp: $(_pid_alive "${RSP_PID_FILE}" robot_state_publisher && echo running || echo stopped)"
    echo "static_tf: $(_pid_alive "${STATIC_PID_FILE}" static_transform_publisher && echo running || echo stopped)"
    echo "rviz: $(_pid_alive "${RVIZ_PID_FILE}" rviz2 && echo running || echo stopped)"
    if [[ -x "${FK_SCRIPT}" ]]; then
        bash "${FK_SCRIPT}" status || true
    fi
}

case "${1:-status}" in
    start) cmd_start ;;
    viz) cmd_viz ;;
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
