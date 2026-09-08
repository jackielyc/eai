#!/usr/bin/env bash
# 在宿主机本地运行 viewer（不进 Docker）
#
# 对照: run_in_docker.sh（ROS/camera 在容器内时必须用 Docker 版，否则 SHM 图像传不出）
# 优先 Humble（RoboStack conda → /opt/ros/humble），否则回退 Jazzy。
#
# 用法:
#   bash run_local.sh
#   bash run_local.sh --help
#
# 环境变量（可选）:
#   PSIBOT_HOME              默认 /home/psibot
#   ROS_HUMBLE_CONDA         RoboStack Humble 环境路径
#   ROS_SETUP_BASH           覆盖 ROS setup.bash（跳过自动探测）
#   A2D_SCRIPTS_DIR          a2d scripts_pack 路径
#   A2D_SDK_HOME             默认 ${PSIBOT_HOME}/a2d_sdk（含 FastDDS xml）
#   PYTHON                   覆盖解释器；Humble conda 时默认用该环境 python
#   LOCAL_QWEN_CTL_PORT      默认 18101
#   REMOTE_QWEN_CTL_PORT     默认 18103
#   SKIP_HOSTCTL=1           跳过 Qwen hostctl 启动
#   SKIP_PIP=1               跳过 pip 依赖安装/检查
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(readlink -f "${SCRIPT_DIR}")"
PSIBOT_HOME="${PSIBOT_HOME:-/home/psibot}"
A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR:-${PSIBOT_HOME}/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts}"
A2D_SDK_HOME="${A2D_SDK_HOME:-${PSIBOT_HOME}/a2d_sdk}"
HOST_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_host.xml"
A2D_DDS_XML="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"
ROS_HUMBLE_CONDA="${ROS_HUMBLE_CONDA:-/share_data/projects/mahjong/share/personal/liyichao/envs/ros-humble}"
USER_PYTHON="${PYTHON:-}"
ROS_MODE=""  # conda_humble | system_humble | system_jazzy

if [[ ! -f "${EAI_DIR}/show_camera_topics.py" ]]; then
    echo "错误: 未找到 ${EAI_DIR}/show_camera_topics.py" >&2
    exit 1
fi
if [[ ! -d "${EAI_DIR}/images" ]]; then
    echo "警告: 未找到 ${EAI_DIR}/images（场景预览图可能缺失）" >&2
fi

echo ">>> 本地模式（非 Docker）"
echo ">>> 工作目录: ${EAI_DIR}"
echo ">>> DISPLAY=${DISPLAY:-:0}"

# ---------------------------------------------------------------------------
# ROS / a2d workspace（Humble 优先）
# ---------------------------------------------------------------------------
ROS_SETUP=""
if [[ -n "${ROS_SETUP_BASH:-}" && -f "${ROS_SETUP_BASH}" ]]; then
    ROS_SETUP="${ROS_SETUP_BASH}"
    if [[ "${ROS_SETUP}" == *humble* ]]; then
        ROS_MODE="system_humble"
    elif [[ "${ROS_SETUP}" == *jazzy* ]]; then
        ROS_MODE="system_jazzy"
    else
        ROS_MODE="custom"
    fi
elif [[ -x "${ROS_HUMBLE_CONDA}/bin/python" && -f "${ROS_HUMBLE_CONDA}/setup.bash" ]]; then
    # RoboStack: 先激活 conda 前缀，再用其 python（rclpy 绑定在该解释器上）
    export CONDA_PREFIX="${ROS_HUMBLE_CONDA}"
    export PATH="${CONDA_PREFIX}/bin${PATH:+:${PATH}}"
    ROS_SETUP="${CONDA_PREFIX}/setup.bash"
    ROS_MODE="conda_humble"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
    ROS_SETUP=/opt/ros/humble/setup.bash
    ROS_MODE="system_humble"
elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    ROS_SETUP=/opt/ros/jazzy/setup.bash
    ROS_MODE="system_jazzy"
fi

if [[ -z "${ROS_SETUP}" ]]; then
    echo "错误: 未找到可用 ROS2（RoboStack Humble /opt/ros/humble /opt/ros/jazzy）" >&2
    echo "  若 ROS/camera 只在 Docker 内，请改用: bash run_in_docker.sh" >&2
    exit 1
fi

set +u
# setup.bash 依赖相对路径；须在 bash 下 source（勿用 zsh 直接 source）
# shellcheck disable=SC1090
source "${ROS_SETUP}"
echo ">>> sourced ${ROS_SETUP} (ROS_DISTRO=${ROS_DISTRO:-?} mode=${ROS_MODE})"
if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/psi/rt/a2d-tele/install/setup.bash
    echo ">>> sourced /opt/psi/rt/a2d-tele/install/setup.bash"
fi
WORKSPACE_INSTALL="${PSIBOT_HOME}/workspace_liyichao/install/setup.bash"
if [[ -f "${WORKSPACE_INSTALL}" ]]; then
    # shellcheck disable=SC1091
    source "${WORKSPACE_INSTALL}"
    echo ">>> sourced ${WORKSPACE_INSTALL}"
fi
set -e

# Python：Humble conda 必须用环境内解释器；否则尊重 PYTHON / 系统 3.10
if [[ -n "${USER_PYTHON}" ]]; then
    PYTHON="${USER_PYTHON}"
elif [[ "${ROS_MODE}" == "conda_humble" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x /usr/bin/python3.10 ]]; then
    PYTHON=/usr/bin/python3.10
else
    PYTHON="$(command -v python3)"
fi
echo ">>> Python: ${PYTHON}"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export A2D_SCRIPTS_DIR
export PSIBOT_HOME
export HOME="${HOME:-${PSIBOT_HOME}}"
export LOCAL_QWEN_CTL_URL="http://127.0.0.1:${LOCAL_QWEN_CTL_PORT:-18101}"
export LOCAL_QWEN_API_BASE="http://127.0.0.1:${LOCAL_QWEN_PORT:-8100}/v1"
export REMOTE_QWEN_CTL_URL="http://127.0.0.1:${REMOTE_QWEN_CTL_PORT:-18103}"

if [[ -f "${HOST_DDS_XML}" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${HOST_DDS_XML}"
    echo ">>> 宿主机 DDS: ${HOST_DDS_XML}"
elif [[ -f "${A2D_DDS_XML}" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${A2D_DDS_XML}"
    echo ">>> A2D DDS: ${A2D_DDS_XML}"
else
    echo ">>> 使用默认 DDS 配置"
fi
echo ">>> ROS_DOMAIN_ID=${ROS_DOMAIN_ID}  ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY}"

ros2 daemon stop >/dev/null 2>&1 || true
ros2 daemon start >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# Locale / IME（本机直接用会话 dbus，无需 Docker 的 abstract 代理）
# ---------------------------------------------------------------------------
if locale -a 2>/dev/null | grep -qi 'zh_CN\.utf8\|zh_CN\.UTF-8'; then
    export LANG="${LANG:-zh_CN.UTF-8}"
    export LC_ALL="${LC_ALL:-zh_CN.UTF-8}"
else
    export LANG="${LANG:-C.UTF-8}"
    export LC_ALL="${LC_ALL:-C.UTF-8}"
fi
export QT_X11_NO_MITSHM=1
export QT_IM_MODULE="${QT_IM_MODULE:-fcitx}"
export GTK_IM_MODULE="${GTK_IM_MODULE:-fcitx}"
export XMODIFIERS="${XMODIFIERS:-@im=fcitx}"
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus" ]]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus"
fi

ensure_fcitx_qt_plugin() {
    local dst_dir="${SCRIPT_DIR}/qt_plugins/platforminputcontexts"
    local dst="${dst_dir}/libfcitxplatforminputcontextplugin.so"
    local src="/usr/lib/x86_64-linux-gnu/qt5/plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so"
    mkdir -p "${dst_dir}"
    if [[ -f "${src}" && ( ! -f "${dst}" || "$(stat -c%s "${src}" 2>/dev/null)" != "$(stat -c%s "${dst}" 2>/dev/null)" ) ]]; then
        cp -f "${src}" "${dst}"
        echo ">>> 已复制 fcitx Qt 插件到 ${dst}"
    fi
}
ensure_fcitx_qt_plugin

# ---------------------------------------------------------------------------
# Qwen hostctl（与 run_in_docker.sh 宿主机侧一致）
# ---------------------------------------------------------------------------
ensure_local_qwen_hostctl() {
    local ctl_py="${SCRIPT_DIR}/local_qwen_hostctl.py"
    local ctl_log="${SCRIPT_DIR}/log/local_qwen_hostctl.log"
    local ctl_port="${LOCAL_QWEN_CTL_PORT:-18101}"
    mkdir -p "${SCRIPT_DIR}/log"
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        echo ">>> 本地 Qwen hostctl 已在运行 (:${ctl_port})"
        return 0
    fi
    if [[ ! -f "${ctl_py}" ]]; then
        echo "警告: 未找到 ${ctl_py}，测试 Tab 启动推理服务可能失败" >&2
        return 0
    fi
    echo ">>> 启动本地 Qwen hostctl (127.0.0.1:${ctl_port})"
    nohup "${PYTHON}" "${ctl_py}" >>"${ctl_log}" 2>&1 &
    disown || true
    sleep 0.4
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        echo ">>> hostctl 就绪。测试 Tab 可点「启动推理服务」"
    else
        echo "警告: hostctl 未能就绪，可手动: ${PYTHON} ${ctl_py}" >&2
    fi
}

remote_hostctl_has_list_models() {
    local port="$1"
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:${port}/list_models?host=psi_motus_2_for_liyichao" 2>/dev/null || echo 000)"
    [[ "${code}" == "200" ]]
}

stop_remote_qwen_hostctl() {
    local ctl_port="$1"
    pkill -f "remote_qwen_hostctl.py" 2>/dev/null || true
    fuser -k "${ctl_port}/tcp" 2>/dev/null || true
    sleep 0.3
}

ensure_remote_qwen_hostctl() {
    local ctl_py="${SCRIPT_DIR}/remote_qwen_hostctl.py"
    local ctl_log="${SCRIPT_DIR}/log/remote_qwen_hostctl.log"
    local ctl_port="${REMOTE_QWEN_CTL_PORT:-18103}"
    mkdir -p "${SCRIPT_DIR}/log"
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        if remote_hostctl_has_list_models "${ctl_port}"; then
            echo ">>> 远程 Qwen hostctl 已在运行 (:${ctl_port})"
            return 0
        fi
        echo ">>> 远程 Qwen hostctl 版本过旧（缺少 list_models），重启…"
        stop_remote_qwen_hostctl "${ctl_port}"
    fi
    if [[ ! -f "${ctl_py}" ]]; then
        echo "警告: 未找到 ${ctl_py}" >&2
        return 0
    fi
    echo ">>> 启动远程 Qwen hostctl (127.0.0.1:${ctl_port})"
    nohup "${PYTHON}" "${ctl_py}" >>"${ctl_log}" 2>&1 &
    disown || true
    sleep 0.4
    if curl -fsS --max-time 1 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        echo ">>> remote hostctl 就绪"
    else
        echo "警告: remote hostctl 未能就绪，可手动: ${PYTHON} ${ctl_py}" >&2
    fi
}

cleanup_stale_remote_qwen_tunnels() {
    if curl -fsS --max-time 1 "http://127.0.0.1:18100/health" >/dev/null 2>&1; then
        echo ">>> 远程 Qwen 隧道已可用 (:18100)"
        return 0
    fi
    if curl -fsS --max-time 1 "http://127.0.0.1:18102/health" >/dev/null 2>&1; then
        echo ">>> 远程 Qwen 隧道已可用 (:18102)"
        return 0
    fi
    echo ">>> 清理无效 SSH 隧道进程"
    pkill -f 'ssh .* -L 18100:127.0.0.1:8100' 2>/dev/null || true
    pkill -f 'ssh .* -L 18102:127.0.0.1:8100' 2>/dev/null || true
    fuser -k 18100/tcp 2>/dev/null || true
    fuser -k 18102/tcp 2>/dev/null || true
}

if [[ "${SKIP_HOSTCTL:-0}" != "1" ]]; then
    ensure_local_qwen_hostctl
    ensure_remote_qwen_hostctl
    cleanup_stale_remote_qwen_tunnels
fi

# ---------------------------------------------------------------------------
# Python / Qt 依赖
# ---------------------------------------------------------------------------
if [[ "${SKIP_PIP:-0}" != "1" ]]; then
    pip_flags=(-q)
    # conda 环境装进 prefix；系统 python 用 --user
    if [[ "${ROS_MODE}" != "conda_humble" ]]; then
        pip_flags+=(--user)
    fi
    # 国内源加速（可用 PIP_INDEX_URL 覆盖）
    pip_index="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    pip_flags+=(-i "${pip_index}" --trusted-host pypi.tuna.tsinghua.edu.cn)
    "${PYTHON}" -m pip install "${pip_flags[@]}" \
        'PyQt5==5.15.10' \
        'opencv-python-headless==4.10.0.84' \
        'numpy>=1.23.5,<2.0.0' \
        'pyqtgraph==0.13.7' \
        'PyOpenGL==3.1.7' 2>/dev/null || true
    "${PYTHON}" -m pip uninstall -y opencv-python 2>/dev/null || true
fi

if ! "${PYTHON}" - <<'PY' 2>/dev/null; then
import numpy
import rclpy
from cv_bridge import CvBridge
assert numpy.__version__.startswith("1.")
PY
    echo "错误: 依赖未就绪（当前 Python=${PYTHON}）。" >&2
    if [[ "${ROS_MODE}" == "conda_humble" ]]; then
        echo "  可重试: ${PYTHON} -m pip install PyQt5 opencv-python-headless pyqtgraph PyOpenGL" >&2
    else
        echo "  请先执行: bash install.sh" >&2
    fi
    exit 1
fi

# 安装 workspace 内 fcitx 插件到当前 PyQt5
"${PYTHON}" - <<PY
import pathlib, shutil
src = pathlib.Path("${EAI_DIR}/qt_plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so")
if not src.is_file():
    raise SystemExit(0)
import PyQt5
dst_dir = pathlib.Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins" / "platforminputcontexts"
dst_dir.mkdir(parents=True, exist_ok=True)
dst = dst_dir / src.name
if (not dst.exists()) or dst.stat().st_size != src.stat().st_size:
    shutil.copy2(src, dst)
    print(f">>> 已安装 fcitx Qt 插件到 {dst}")
PY

unset QT_PLUGIN_PATH
PYQT_QT_LIB="$("${PYTHON}" -c 'import PyQt5, pathlib; print(pathlib.Path(PyQt5.__file__).resolve().parent / "Qt5" / "lib")' 2>/dev/null || true)"
if [[ -n "${PYQT_QT_LIB}" && -d "${PYQT_QT_LIB}" ]]; then
    export LD_LIBRARY_PATH="${PYQT_QT_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo ">>> PyQt5 Qt lib: ${PYQT_QT_LIB}"
fi
QT5_PLATFORMS="$("${PYTHON}" -c 'import PyQt5, os; print(os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms"))' 2>/dev/null || true)"
if [[ -n "${QT5_PLATFORMS}" && -d "${QT5_PLATFORMS}" ]]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="${QT5_PLATFORMS}"
fi

# ---------------------------------------------------------------------------
# 预检：topic / 是否误连 Docker SHM
# ---------------------------------------------------------------------------
echo ">>> 预检 topic..."
topic_count="$("${PYTHON}" - <<'PY' 2>/dev/null || echo 0
import rclpy
rclpy.init()
from rclpy.node import Node
n = Node("preflight_check")
print(len(dict(n.get_topic_names_and_types())))
n.destroy_node()
rclpy.shutdown()
PY
)"
camera_count="$("${PYTHON}" - <<'PY' 2>/dev/null || echo 0
import rclpy
rclpy.init()
from rclpy.node import Node
n = Node("preflight_cam")
topics = [t for t in dict(n.get_topic_names_and_types()) if t.startswith("/camera")]
print(len(topics))
n.destroy_node()
rclpy.shutdown()
PY
)"
camera_hz="$({ timeout 3 ros2 topic hz /camera/head_color 2>&1 | grep -c "average rate" || true; })"
echo ">>> 预检: ${topic_count} 个 topic，${camera_count} 个 /camera*"

if [[ "${topic_count}" -ge 10 && "${camera_hz}" -eq 0 ]]; then
    echo ""
    echo ">>> ⚠️  检测到 topic 但无图像数据 — ROS 很可能在 Docker 内运行" >&2
    echo ">>>     Docker 使用 SHM 共享内存，宿主机无法接收图像" >&2
    echo ">>>     请改用: bash run_in_docker.sh" >&2
    echo ""
fi

echo ">>> IME QT_IM_MODULE=${QT_IM_MODULE} DBUS=${DBUS_SESSION_BUS_ADDRESS:-<unset>}"
exec "${PYTHON}" "${EAI_DIR}/show_camera_topics.py" "$@"
