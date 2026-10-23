#!/usr/bin/env bash
# 在宿主机本地启动 viewer 界面（不进 Docker）。
#
# 本脚本只负责：配好 ROS/Qt 环境 → 启动 show_camera_boot.py。
# 不自动启手驱动、中继、SHM chmod、hostctl 等；那些请自行跑对应 scripts。
#
# 用法:
#   bash run_local.sh
#   RUN_CHECKS=1 bash run_local.sh   # 可选：依赖/hostctl 等校验
#
# 环境变量（可选）:
#   PSIBOT_HOME / ROS_HUMBLE_CONDA / ROS_SETUP_BASH / PYTHON
#   A2D_SCRIPTS_DIR / A2D_SDK_HOME
#   FORCE_ROS_SOURCE=1 / RUN_CHECKS=1 / SKIP_HOSTCTL=1
#   FORCE_DEPS_CHECK=1 / FORCE_PIP=1 / PREFLIGHT=1 / ROS_RESTART_DAEMON=1
#   ALIGN_DOCKER_CAMERA=0  不因 Docker 相机改 Domain/RMW（默认仍对齐 env，但不跑副作用脚本）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# macOS BSD readlink 无可靠 -f；pwd -P 两端都可用
EAI_DIR="$(cd "${SCRIPT_DIR}" && pwd -P)"
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
IS_DARWIN=0
[[ "${UNAME_S}" == "Darwin" ]] && IS_DARWIN=1

if [[ -z "${PSIBOT_HOME:-}" ]]; then
    if [[ "${IS_DARWIN}" -eq 1 ]]; then
        PSIBOT_HOME="${HOME}"
    else
        PSIBOT_HOME="/home/psibot"
    fi
fi
A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR:-${PSIBOT_HOME}/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts}"
A2D_SDK_HOME="${A2D_SDK_HOME:-${PSIBOT_HOME}/a2d_sdk}"
HOST_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_host.xml"
MAC_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_mac.xml"
A2D_DDS_XML="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"
EAI_A2D_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_a2d.xml"
# Linux 默认保持原 share_data 路径（不做存在性探测，避免拖慢启动）
_LINUX_ROS_CONDA_DEFAULT="/share_data/projects/mahjong/share/personal/liyichao/envs/ros-humble"
USER_PYTHON="${PYTHON:-}"
ROS_MODE=""

# BSD date 的 +%s%3N 会产出 "….3N" 且 exit 0；只认纯 13 位毫秒时间戳
_now_epoch() {
    local t
    t="$(date +%s%3N 2>/dev/null || true)"
    if [[ "${t}" =~ ^[0-9]{13}$ ]]; then
        printf '%s\n' "${t}"
    else
        date +%s
    fi
}
_t0="$(_now_epoch)"

_log_elapsed() {
    local now
    now="$(_now_epoch)"
    if [[ "${#_t0}" -ge 13 && "${#now}" -ge 13 ]]; then
        echo ">>> 启动耗时 $((now - _t0)) ms"
    else
        echo ">>> 启动耗时 $((now - _t0)) s"
    fi
}

_prepend_lib_path() {
    local extra="$1"
    [[ -n "${extra}" ]] || return 0
    if [[ "${IS_DARWIN}" -eq 1 ]]; then
        export DYLD_FALLBACK_LIBRARY_PATH="${extra}${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"
    else
        export LD_LIBRARY_PATH="${extra}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

_conda_python_site_packages() {
    # $2=fast：不执行 python（Linux share_data 冷启动要快）
    local prefix="$1"
    local mode="${2:-}"
    local py ver major minor
    if [[ "${mode}" == "fast" ]]; then
        printf '%s\n' \
            "${prefix}/lib/python311/site-packages" \
            "${prefix}/lib/python3.11/site-packages"
        return 0
    fi
    py="${prefix}/bin/python"
    if [[ -x "${py}" ]]; then
        ver="$("${py}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        if [[ -n "${ver}" ]]; then
            major="${ver%%.*}"
            minor="${ver#*.}"
            printf '%s\n' \
                "${prefix}/lib/python${major}${minor}/site-packages" \
                "${prefix}/lib/python${ver}/site-packages"
            return 0
        fi
    fi
    printf '%s\n' \
        "${prefix}/lib/python311/site-packages" \
        "${prefix}/lib/python3.11/site-packages"
}

_pick_mac_ros_conda() {
    # 优先：显式 ROS_HUMBLE_CONDA → 当前 CONDA_PREFIX → 常见本机 env
    local c
    for c in \
        "${ROS_HUMBLE_CONDA:-}" \
        "${CONDA_PREFIX:-}" \
        "/opt/homebrew/Caskroom/miniconda/base/envs/ros2-jazzy" \
        "/opt/homebrew/Caskroom/miniconda/base/envs/ros2-humble" \
        "${HOME}/miniconda3/envs/ros2-jazzy" \
        "${HOME}/miniconda3/envs/ros2-humble" \
        "${HOME}/mambaforge/envs/ros2-jazzy" \
        "${HOME}/mambaforge/envs/ros2-humble" \
        "${HOME}/anaconda3/envs/ros2-jazzy" \
        "${HOME}/anaconda3/envs/ros2-humble"
    do
        [[ -n "${c}" && -x "${c}/bin/python" ]] || continue
        if [[ -d "${c}/lib/python3.12/site-packages/PyQt5" \
           || -d "${c}/lib/python3.11/site-packages/PyQt5" \
           || -d "${c}/lib/python3.10/site-packages/PyQt5" ]]; then
            printf '%s\n' "${c}"
            return 0
        fi
    done
    for c in \
        "${ROS_HUMBLE_CONDA:-}" \
        "${CONDA_PREFIX:-}" \
        "/opt/homebrew/Caskroom/miniconda/base/envs/ros2-jazzy" \
        "/opt/homebrew/Caskroom/miniconda/base/envs/ros2-humble" \
        "${HOME}/miniconda3/envs/ros2-jazzy" \
        "${HOME}/miniconda3/envs/ros2-humble"
    do
        [[ -n "${c}" && -x "${c}/bin/python" ]] || continue
        printf '%s\n' "${c}"
        return 0
    done
    return 1
}

_apply_conda_humble_fast_env() {
    local prefix="$1"
    local distro="${2:-humble}"
    local site_mode="${3:-fast}"
    local os_tag="linux"
    local pp="" site
    [[ "${IS_DARWIN}" -eq 1 ]] && os_tag="osx"
    export CONDA_PREFIX="${prefix}"
    export PATH="${prefix}/bin${PATH:+:${PATH}}"
    export AMENT_PREFIX_PATH="${prefix}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
    export ROS_ETC_DIR="${prefix}/etc/ros"
    export ROS_DISTRO="${distro}"
    export ROS_VERSION=2
    export ROS_PYTHON_VERSION=3
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export ROS_OS_OVERRIDE="${ROS_OS_OVERRIDE:-conda:${os_tag}}"
    export GSETTINGS_SCHEMA_DIR="${prefix}/share/glib-2.0/schemas${GSETTINGS_SCHEMA_DIR:+:${GSETTINGS_SCHEMA_DIR}}"
    _prepend_lib_path "${prefix}/lib"
    while IFS= read -r site; do
        [[ -n "${site}" ]] || continue
        if [[ -z "${pp}" ]]; then
            pp="${site}"
        else
            pp="${pp}:${site}"
        fi
    done < <(_conda_python_site_packages "${prefix}" "${site_mode}")
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
        _prepend_lib_path "${qt_lib}"
    fi
    if [[ -n "${qt_platforms}" ]]; then
        export QT_QPA_PLATFORM_PLUGIN_PATH="${qt_platforms}"
    fi
}

_resolve_conda_pyqt_root() {
    local prefix="$1"
    local site root
    while IFS= read -r site; do
        for root in \
            "${site}/PyQt5/Qt5" \
            "${site}/PyQt5/Qt"
        do
            if [[ -d "${root}/plugins/platforms" ]]; then
                printf '%s\n' "${root}"
                return 0
            fi
        done
    done < <(_conda_python_site_packages "${prefix}")
    return 1
}

_apply_qt_from_python() {
    # 系统 / 非 conda PyQt：用当前 PYTHON 找 platforms；清掉 cv2 插件劫持
    local qt_plat qt_lib
    unset QT_PLUGIN_PATH
    if [[ -n "${QT_QPA_PLATFORM_PLUGIN_PATH:-}" && "${QT_QPA_PLATFORM_PLUGIN_PATH}" == *"/cv2/"* ]]; then
        unset QT_QPA_PLATFORM_PLUGIN_PATH
    fi
    [[ -z "${QT_QPA_PLATFORM_PLUGIN_PATH:-}" ]] || return 0
    qt_plat="$("${PYTHON}" - <<'PY' 2>/dev/null || true
import glob
import os

try:
    import PyQt5
    root = os.path.dirname(PyQt5.__file__)
    for rel in ("Qt5/plugins/platforms", "Qt/plugins/platforms"):
        p = os.path.join(root, rel)
        if os.path.isdir(p):
            print(p)
            raise SystemExit
except SystemExit:
    raise
except Exception:
    pass
# Debian/Ubuntu 系统包：插件在 /usr/lib/<triplet>/qt5/plugins
for p in glob.glob("/usr/lib/*/qt5/plugins/platforms"):
    if os.path.isdir(p):
        print(p)
        break
PY
)"
    if [[ -n "${qt_plat}" ]]; then
        export QT_QPA_PLATFORM_PLUGIN_PATH="${qt_plat}"
        qt_lib="$(dirname "$(dirname "${qt_plat}")")/lib"
        if [[ -d "${qt_lib}" && "${qt_lib}" == *"/PyQt5/"* ]]; then
            _prepend_lib_path "${qt_lib}"
        fi
    fi
}

_pick_linux_ros_conda() {
    # 显式 ROS_HUMBLE_CONDA → 当前 CONDA_PREFIX → share_data 默认 → 常见本机 env
    local c
    for c in \
        "${ROS_HUMBLE_CONDA:-}" \
        "${CONDA_PREFIX:-}" \
        "${_LINUX_ROS_CONDA_DEFAULT}" \
        "${HOME}/miniconda3/envs/ros-humble" \
        "${HOME}/mambaforge/envs/ros-humble" \
        "${HOME}/anaconda3/envs/ros-humble"
    do
        [[ -n "${c}" && -x "${c}/bin/python" ]] || continue
        printf '%s\n' "${c}"
        return 0
    done
    # 默认路径不做存在性探测（share_data 冷启动要快）；交给后续报错
    if [[ -n "${ROS_HUMBLE_CONDA:-}" ]]; then
        printf '%s\n' "${ROS_HUMBLE_CONDA}"
        return 0
    fi
    printf '%s\n' "${_LINUX_ROS_CONDA_DEFAULT}"
    return 0
}

# ---------------------------------------------------------------------------
# ROS：Linux 优先系统 humble，否则 RoboStack conda；
#      macOS 自动探测本机 conda ROS（jazzy/humble）
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
elif [[ "${IS_DARWIN}" -eq 1 ]]; then
    if [[ -z "${ROS_HUMBLE_CONDA:-}" ]]; then
        ROS_HUMBLE_CONDA="$(_pick_mac_ros_conda)" || {
            echo "错误: macOS 未找到可用的 conda ROS 环境（需含 bin/python）。" >&2
            echo "请设置 ROS_HUMBLE_CONDA / CONDA_PREFIX / PYTHON，或安装 ros2-jazzy / ros2-humble。" >&2
            exit 1
        }
    fi
    export CONDA_PREFIX="${ROS_HUMBLE_CONDA}"
    ROS_SETUP="${CONDA_PREFIX}/setup.bash"
    if [[ "${CONDA_PREFIX}" == *jazzy* ]]; then
        ROS_MODE="conda_jazzy"
    else
        ROS_MODE="conda_humble"
    fi
else
    # Linux：显式 ROS_HUMBLE_CONDA 走 conda；否则优先 /opt/ros/humble + python3.10
    if [[ -n "${ROS_HUMBLE_CONDA:-}" ]]; then
        export CONDA_PREFIX="${ROS_HUMBLE_CONDA}"
        ROS_SETUP="${CONDA_PREFIX}/setup.bash"
        ROS_MODE="conda_humble"
    elif [[ -f /opt/ros/humble/setup.bash && -x /usr/bin/python3.10 ]]; then
        ROS_SETUP="/opt/ros/humble/setup.bash"
        ROS_MODE="system_humble"
        USER_PYTHON="${USER_PYTHON:-/usr/bin/python3.10}"
    else
        ROS_HUMBLE_CONDA="$(_pick_linux_ros_conda)"
        export CONDA_PREFIX="${ROS_HUMBLE_CONDA}"
        ROS_SETUP="${CONDA_PREFIX}/setup.bash"
        ROS_MODE="conda_humble"
    fi
fi

set +u
if [[ "${ROS_MODE}" == "conda_humble" && "${IS_DARWIN}" -eq 0 && "${FORCE_ROS_SOURCE:-0}" != "1" ]]; then
    # Linux RoboStack：跳过 source，直接拼环境（share_data 上 source 很慢）
    _apply_conda_humble_fast_env "${CONDA_PREFIX}" humble
elif [[ "${ROS_MODE}" == "conda_jazzy" || "${ROS_MODE}" == "conda_humble" ]]; then
    # macOS / 强制 source：走 setup.bash（本机盘很快，且能正确设置 dylib）
    if [[ -f "${ROS_SETUP}" ]]; then
        # shellcheck disable=SC1090
        source "${ROS_SETUP}"
    else
        # macOS 回退：探测真实 python 版本（本机盘，可接受一次 python -c）
        _apply_conda_humble_fast_env "${CONDA_PREFIX}" \
            "$([[ "${ROS_MODE}" == "conda_jazzy" ]] && echo jazzy || echo humble)" \
            probe
    fi
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
elif [[ "${ROS_MODE}" == "conda_humble" || "${ROS_MODE}" == "conda_jazzy" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x /usr/bin/python3.10 ]]; then
    PYTHON=/usr/bin/python3.10
else
    PYTHON="$(command -v python3)"
fi

# Linux 默认 FastDDS；macOS 默认 CycloneDDS —
# RoboStack FastDDS 在 macOS 上 destroy_subscription 会永久卡住，Qt 主线程抢不到 GIL → 界面假死
#
# 相机在 Docker（a2d-tele）时：仅对齐 Domain/RMW/DDS XML 环境变量，方便 UI 收图。
# 不在此 chmod SHM、不启驱动（见 scripts/enable_host_ros2_camera.sh / start_robot_hand_drivers.sh）。
_DOCKER_CAM_CONTAINER="${A2D_DOCKER_CONTAINER:-a2d-tele-release-2-1-0rc3-latest}"
_TELE_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_tele.xml"
_SHM_UDP_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_shm_udp.xml"
_ALIGN_DOCKER_CAM=0
# 永远不用 Docker：宿主机 forwarder（相机）+ host_arm_hand_bridge（使能）
if [[ "${IS_DARWIN}" -eq 1 ]]; then
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
    export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
else
    # 强制对齐宿主机 forwarder。Cursor/shell 若残留 cyclone+domain161，
    # ${VAR:-default} 会原样继承 → UI 收不到 /camera/*。
    # 需要保留外层 ROS 环境时：EAI_KEEP_ROS_ENV=1 bash run_local.sh
    if [[ "${EAI_KEEP_ROS_ENV:-0}" != "1" ]]; then
        if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_fastrtps_cpp" \
           || "${ROS_DOMAIN_ID:-}" != "0" ]]; then
            echo ">>> 覆盖继承的 ROS 环境 → FastDDS domain0（与相机 forwarder 一致；原 RMW=${RMW_IMPLEMENTATION:-?} domain=${ROS_DOMAIN_ID:-?}）" >&2
        fi
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
        export ROS_DOMAIN_ID=0
        export ROS_LOCALHOST_ONLY=1
        unset ROS_AUTOMATIC_DISCOVERY_RANGE 2>/dev/null || true
    else
        export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
        export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
        export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
    fi
fi
export A2D_SCRIPTS_DIR
export PSIBOT_HOME
export HOME="${HOME:-${PSIBOT_HOME}}"
export LOCAL_QWEN_CTL_URL="http://127.0.0.1:${LOCAL_QWEN_CTL_PORT:-18101}"
# EmbodiedEvo Coder 默认走 vLLM :8000（Responses）。HF worker 仍用 LOCAL_QWEN_PORT=8100。
export LOCAL_QWEN_API_BASE="${LOCAL_QWEN_API_BASE:-http://127.0.0.1:${LOCAL_VLLM_PORT:-${LOCAL_API_PORT:-8000}}/v1}"
export REMOTE_QWEN_CTL_URL="http://127.0.0.1:${REMOTE_QWEN_CTL_PORT:-18103}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-${LANG}}"
export QT_X11_NO_MITSHM=1

# 大文件在 share_data 上冷编译很慢；pycache 写到本地盘
if [[ -z "${PYTHONPYCACHEPREFIX:-}" ]]; then
    export PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME:-${HOME}/.cache}/eai_pycache"
    mkdir -p "${PYTHONPYCACHEPREFIX}" 2>/dev/null || true
fi

# DDS XML：仅 FastDDS 需要。macOS 默认 Cyclone，不挂 Linux SHM host 配置
# 失效路径会触发 [XMLPARSER Error] realpath failed → loadDefaultXMLFile
if [[ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" && ! -f "${FASTRTPS_DEFAULT_PROFILES_FILE}" ]]; then
    echo "警告: FASTRTPS_DEFAULT_PROFILES_FILE 无效 (${FASTRTPS_DEFAULT_PROFILES_FILE})，已 unset" >&2
    unset FASTRTPS_DEFAULT_PROFILES_FILE
fi
if [[ "${RMW_IMPLEMENTATION}" == *fastrtps* ]]; then
    if [[ -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
        if [[ "${IS_DARWIN}" -eq 1 ]]; then
            if [[ -f "${MAC_DDS_XML}" ]]; then
                export FASTRTPS_DEFAULT_PROFILES_FILE="${MAC_DDS_XML}"
            fi
        else
            # 无 Docker：shm+UDP（与宿主机 forwarder / 使能桥一致）
            if [[ -f "${_SHM_UDP_DDS_XML}" ]]; then
                export FASTRTPS_DEFAULT_PROFILES_FILE="${_SHM_UDP_DDS_XML}"
            elif [[ -f "${_TELE_DDS_XML}" ]]; then
                export FASTRTPS_DEFAULT_PROFILES_FILE="${_TELE_DDS_XML}"
            elif [[ -f "${EAI_A2D_DDS_XML}" ]]; then
                export FASTRTPS_DEFAULT_PROFILES_FILE="${EAI_A2D_DDS_XML}"
            elif [[ -f "${HOST_DDS_XML}" ]]; then
                export FASTRTPS_DEFAULT_PROFILES_FILE="${HOST_DDS_XML}"
            fi
        fi
    fi
fi

# macOS：强制 Cocoa，避免 XQuartz DISPLAY 把平台插件带偏
if [[ "${IS_DARWIN}" -eq 1 ]]; then
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-cocoa}"
fi

# Qt：Linux conda humble 快路径直接拼；其余 conda 探测；系统 ROS 用当前 Python/系统包
if [[ "${ROS_MODE}" == "conda_humble" && "${IS_DARWIN}" -eq 0 ]]; then
    _conda_pyqt_root="${CONDA_PREFIX}/lib/python3.11/site-packages/PyQt5/Qt5"
    _apply_qt_paths "${_conda_pyqt_root}/lib" "${_conda_pyqt_root}/plugins/platforms"
elif [[ "${ROS_MODE}" == "conda_humble" || "${ROS_MODE}" == "conda_jazzy" ]]; then
    if _conda_pyqt_root="$(_resolve_conda_pyqt_root "${CONDA_PREFIX}")"; then
        _apply_qt_paths "${_conda_pyqt_root}/lib" "${_conda_pyqt_root}/plugins/platforms"
    fi
else
    _apply_qt_from_python
fi

echo ">>> 本地模式 DISPLAY=${DISPLAY:-:0} Python=${PYTHON} ROS=${ROS_DISTRO:-?} (${ROS_MODE}) RMW=${RMW_IMPLEMENTATION} domain=${ROS_DOMAIN_ID} localhost_only=${ROS_LOCALHOST_ONLY}"
# 注意：不在此启动手驱动 / relay / SHM chmod；只拉 UI。

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
if [[ "${ROS_MODE}" == "conda_humble" && "${IS_DARWIN}" -eq 0 ]]; then
    _conda_pyqt_root="${CONDA_PREFIX}/lib/python3.11/site-packages/PyQt5/Qt5"
    _apply_qt_paths "${_conda_pyqt_root}/lib" "${_conda_pyqt_root}/plugins/platforms"
elif [[ "${ROS_MODE}" == "conda_humble" || "${ROS_MODE}" == "conda_jazzy" ]]; then
    if _conda_pyqt_root="$(_resolve_conda_pyqt_root "${CONDA_PREFIX}")"; then
        _apply_qt_paths "${_conda_pyqt_root}/lib" "${_conda_pyqt_root}/plugins/platforms"
    fi
else
    _apply_qt_from_python
fi
# WebEngine：关 GPU 合成，减少无加速/远程 X 刷屏。勿默认 QT_XCB_GL_INTEGRATION=none
# （会与 AA_ShareOpenGLContexts 冲突，启动时 SIP segfault）。无 GLX 时再显式：
#   export QT_XCB_GL_INTEGRATION=none
if [[ -z "${QTWEBENGINE_CHROMIUM_FLAGS:-}" ]]; then
    export QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu --disable-gpu-compositing --disable-webgl --disable-dev-shm-usage --num-raster-threads=2"
fi
exec "${PYTHON}" "${EAI_DIR}/show_camera_boot.py" "$@"
