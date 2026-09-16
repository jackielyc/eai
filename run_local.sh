#!/usr/bin/env bash
# 在宿主机本地运行 viewer（不进 Docker）
#
# 默认：不做依赖/hostctl/topic 等校验，直接启动界面（尽快）。
# 对照: run_in_docker.sh（ROS/camera 在容器内时用 Docker 版）
#
# 用法:
#   bash run_local.sh
#   RUN_CHECKS=1 bash run_local.sh   # 启用探测/安装/hostctl 等完整校验
#
# 环境变量（可选）:
#   PSIBOT_HOME / ROS_HUMBLE_CONDA / ROS_SETUP_BASH / PYTHON
#   A2D_SCRIPTS_DIR / A2D_SDK_HOME
#   FORCE_ROS_SOURCE=1   conda 也强制 source setup.bash
#   RUN_CHECKS=1         启用校验与 hostctl
#   SKIP_HOSTCTL=1       RUN_CHECKS 时仍跳过 hostctl
#   FORCE_DEPS_CHECK=1 / FORCE_PIP=1 / PREFLIGHT=1 / PREFLIGHT_HZ=1
#   ROS_RESTART_DAEMON=1
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
ROS_MODE=""
_t0="$(date +%s%3N 2>/dev/null || date +%s)"

_log_elapsed() {
    local now
    now="$(date +%s%3N 2>/dev/null || date +%s)"
    if [[ "${#_t0}" -ge 13 && "${#now}" -ge 13 ]]; then
        echo ">>> 启动耗时 $((now - _t0)) ms"
    else
        echo ">>> 启动耗时 $((now - _t0)) s"
    fi
}

_apply_conda_humble_fast_env() {
    local prefix="$1"
    export CONDA_PREFIX="${prefix}"
    export PATH="${prefix}/bin${PATH:+:${PATH}}"
    export AMENT_PREFIX_PATH="${prefix}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
    export ROS_ETC_DIR="${prefix}/etc/ros"
    export ROS_DISTRO=humble
    export ROS_VERSION=2
    export ROS_PYTHON_VERSION=3
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export ROS_OS_OVERRIDE="${ROS_OS_OVERRIDE:-conda:linux}"
    export GSETTINGS_SCHEMA_DIR="${prefix}/share/glib-2.0/schemas${GSETTINGS_SCHEMA_DIR:+:${GSETTINGS_SCHEMA_DIR}}"
    export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    local pp="${prefix}/lib/python311/site-packages:${prefix}/lib/python3.11/site-packages"
    export PYTHONPATH="${pp}${PYTHONPATH:+:${PYTHONPATH}}"
}

_apply_qt_paths() {
    local qt_lib="$1"
    local qt_platforms="$2"
    unset QT_PLUGIN_PATH
    # 避免 opencv-contrib 等把路径指到 cv2/qt/plugins，导致 xcb 加载失败
    if [[ -n "${QT_QPA_PLATFORM_PLUGIN_PATH:-}" && "${QT_QPA_PLATFORM_PLUGIN_PATH}" == *"/cv2/"* ]]; then
        unset QT_QPA_PLATFORM_PLUGIN_PATH
    fi
    if [[ -n "${qt_lib}" ]]; then
        export LD_LIBRARY_PATH="${qt_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    if [[ -n "${qt_platforms}" ]]; then
        export QT_QPA_PLATFORM_PLUGIN_PATH="${qt_platforms}"
    fi
}

# ---------------------------------------------------------------------------
# ROS：默认假定 RoboStack conda；不 stat/探测（加快 share_data）
# ---------------------------------------------------------------------------
if [[ -n "${ROS_SETUP_BASH:-}" ]]; then
    ROS_SETUP="${ROS_SETUP_BASH}"
    if [[ "${ROS_SETUP}" == *humble* ]]; then
        ROS_MODE="system_humble"
    elif [[ "${ROS_SETUP}" == *jazzy* ]]; then
        ROS_MODE="system_jazzy"
    else
        ROS_MODE="custom"
    fi
else
    export CONDA_PREFIX="${ROS_HUMBLE_CONDA}"
    ROS_SETUP="${CONDA_PREFIX}/setup.bash"
    ROS_MODE="conda_humble"
fi

set +u
if [[ "${ROS_MODE}" == "conda_humble" && "${FORCE_ROS_SOURCE:-0}" != "1" ]]; then
    _apply_conda_humble_fast_env "${CONDA_PREFIX}"
else
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
fi
if [[ "${RUN_CHECKS:-0}" == "1" ]]; then
    if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
        # shellcheck disable=SC1091
        source /opt/psi/rt/a2d-tele/install/setup.bash
    fi
    WORKSPACE_INSTALL="${PSIBOT_HOME}/workspace_liyichao/install/setup.bash"
    if [[ -f "${WORKSPACE_INSTALL}" ]]; then
        # shellcheck disable=SC1091
        source "${WORKSPACE_INSTALL}"
    fi
fi
set -e

if [[ -n "${USER_PYTHON}" ]]; then
    PYTHON="${USER_PYTHON}"
elif [[ "${ROS_MODE}" == "conda_humble" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x /usr/bin/python3.10 ]]; then
    PYTHON=/usr/bin/python3.10
else
    PYTHON="$(command -v python3)"
fi

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
# 与 fast-env / setup 一致：若已设置则保留，否则默认 0（可见跨机 topic）
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export A2D_SCRIPTS_DIR
export PSIBOT_HOME
export HOME="${HOME:-${PSIBOT_HOME}}"
export LOCAL_QWEN_CTL_URL="http://127.0.0.1:${LOCAL_QWEN_CTL_PORT:-18101}"
export LOCAL_QWEN_API_BASE="http://127.0.0.1:${LOCAL_QWEN_PORT:-8100}/v1"
export REMOTE_QWEN_CTL_URL="http://127.0.0.1:${REMOTE_QWEN_CTL_PORT:-18103}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-${LANG}}"
export QT_X11_NO_MITSHM=1

if [[ -f "${HOST_DDS_XML}" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${HOST_DDS_XML}"
elif [[ -f "${A2D_DDS_XML}" ]]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="${A2D_DDS_XML}"
fi

# Qt：直接拼路径，不做存在性校验
if [[ "${ROS_MODE}" == "conda_humble" ]]; then
    _conda_pyqt_root="${CONDA_PREFIX}/lib/python3.11/site-packages/PyQt5/Qt5"
    _apply_qt_paths "${_conda_pyqt_root}/lib" "${_conda_pyqt_root}/plugins/platforms"
fi

echo ">>> 本地模式 DISPLAY=${DISPLAY:-:0} Python=${PYTHON} ROS=${ROS_DISTRO:-?} (${ROS_MODE})"

# ---------------------------------------------------------------------------
# 可选校验（默认关闭）
# ---------------------------------------------------------------------------
if [[ "${RUN_CHECKS:-0}" == "1" ]]; then
    CACHE_DIR="${SCRIPT_DIR}/.cache"
    mkdir -p "${CACHE_DIR}"
    DEPS_MARKER="${CACHE_DIR}/deps_ok"

    if [[ "${ROS_RESTART_DAEMON:-0}" == "1" ]]; then
        ros2 daemon stop >/dev/null 2>&1 || true
        ros2 daemon start >/dev/null 2>&1 || true
    fi

    if [[ "${SKIP_HOSTCTL:-0}" != "1" ]]; then
        _port_listening() {
            local port="$1"
            (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
        }
        if ! _port_listening "${LOCAL_QWEN_CTL_PORT:-18101}"; then
            mkdir -p "${SCRIPT_DIR}/log"
            nohup "${PYTHON}" "${SCRIPT_DIR}/local_qwen_hostctl.py" \
                >>"${SCRIPT_DIR}/log/local_qwen_hostctl.log" 2>&1 &
            disown || true
            echo ">>> 已后台启动本地 Qwen hostctl"
        fi
        if ! _port_listening "${REMOTE_QWEN_CTL_PORT:-18103}"; then
            mkdir -p "${SCRIPT_DIR}/log"
            nohup "${PYTHON}" "${SCRIPT_DIR}/remote_qwen_hostctl.py" \
                >>"${SCRIPT_DIR}/log/remote_qwen_hostctl.log" 2>&1 &
            disown || true
            echo ">>> 已后台启动远程 Qwen hostctl"
        fi
    fi

    if [[ "${FORCE_DEPS_CHECK:-0}" == "1" ]] || [[ ! -f "${DEPS_MARKER}" ]]; then
        echo ">>> RUN_CHECKS: 探测 Python 依赖…"
        if ! "${PYTHON}" - <<'PY' 2>/dev/null; then
import numpy
import PyQt5  # noqa: F401
import rclpy  # noqa: F401
from cv_bridge import CvBridge  # noqa: F401
assert numpy.__version__.startswith("1.")
PY
            if [[ "${FORCE_PIP:-0}" == "1" || "${SKIP_PIP:-0}" != "1" ]]; then
                pip_index="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
                "${PYTHON}" -m pip install -q \
                    -i "${pip_index}" --trusted-host pypi.tuna.tsinghua.edu.cn \
                    'PyQt5==5.15.10' \
                    'opencv-python-headless==4.10.0.84' \
                    'numpy>=1.23.5,<2.0.0' \
                    'pyqtgraph==0.13.7' \
                    'PyOpenGL==3.1.7' 2>/dev/null || true
            fi
        fi
        printf '%s\n' "${PYTHON}|${ROS_MODE}|${CONDA_PREFIX:-}" >"${DEPS_MARKER}"
    fi

    if [[ "${PREFLIGHT:-0}" == "1" ]]; then
        echo ">>> 预检 topic..."
        counts="$("${PYTHON}" - <<'PY' 2>/dev/null || echo "0 0"
import rclpy
rclpy.init()
from rclpy.node import Node
n = Node("preflight_check")
names = list(dict(n.get_topic_names_and_types()))
cams = sum(1 for t in names if t.startswith("/camera"))
print(f"{len(names)} {cams}")
n.destroy_node()
rclpy.shutdown()
PY
)"
        echo ">>> 预检: ${counts%% *} topic, ${counts##* } /camera*"
    fi
fi

_log_elapsed
# 启动前再钉一次 PyQt 插件路径（防止先前环境 / OpenCV 污染）
if [[ "${ROS_MODE}" == "conda_humble" ]]; then
    _conda_pyqt_root="${CONDA_PREFIX}/lib/python3.11/site-packages/PyQt5/Qt5"
    _apply_qt_paths "${_conda_pyqt_root}/lib" "${_conda_pyqt_root}/plugins/platforms"
fi
# 远程 X 常无 GLX：提前关掉 WebEngine GPU，避免 ANGLE/GLX 刷屏
if [[ -z "${QTWEBENGINE_CHROMIUM_FLAGS:-}" ]]; then
    export QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu --disable-gpu-compositing --disable-webgl --disable-dev-shm-usage --in-process-gpu"
fi
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
exec "${PYTHON}" "${EAI_DIR}/show_camera_topics.py" "$@"
