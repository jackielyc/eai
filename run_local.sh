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
#   SKIP_PIP=1               强制跳过 pip（默认已是「缺包才装」）
#   FORCE_PIP=1              每次强制 pip install
#   SKIP_DEPS_CHECK=1        跳过 Python 依赖探测（更快；默认有 cache 也会跳）
#   FORCE_DEPS_CHECK=1       强制重新探测依赖并刷新 cache
#   FORCE_ROS_SOURCE=1       强制 source setup.bash（默认 conda 走快路径，不 source）
#   SKIP_PREFLIGHT=1         跳过 topic 预检（默认；更快）
#   PREFLIGHT=1              启用轻量 topic 预检（不含 3s hz）
#   PREFLIGHT_HZ=1           预检时额外跑 ros2 topic hz（约 +3s）
#   ROS_RESTART_DAEMON=1     重启 ros2 daemon（默认不重启）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR="$(readlink -f "${SCRIPT_DIR}")"
CACHE_DIR="${SCRIPT_DIR}/.cache"
PSIBOT_HOME="${PSIBOT_HOME:-/home/psibot}"
A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR:-${PSIBOT_HOME}/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts}"
A2D_SDK_HOME="${A2D_SDK_HOME:-${PSIBOT_HOME}/a2d_sdk}"
HOST_DDS_XML="${SCRIPT_DIR}/dds/fastdds_profiles_host.xml"
A2D_DDS_XML="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"
ROS_HUMBLE_CONDA="${ROS_HUMBLE_CONDA:-/share_data/projects/mahjong/share/personal/liyichao/envs/ros-humble}"
USER_PYTHON="${PYTHON:-}"
ROS_MODE=""  # conda_humble | system_humble | system_jazzy
_t0="$(date +%s%3N 2>/dev/null || date +%s)"
mkdir -p "${CACHE_DIR}"

_log_elapsed() {
    local now
    now="$(date +%s%3N 2>/dev/null || date +%s)"
    if [[ "${#_t0}" -ge 13 && "${#now}" -ge 13 ]]; then
        echo ">>> 启动耗时 $((now - _t0)) ms"
    else
        echo ">>> 启动耗时 $((now - _t0)) s"
    fi
}

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
    # RoboStack: 用环境内 python（rclpy 绑定在该解释器上）
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

# RoboStack 在 share_data 上 source setup.bash 常需数秒（ament 扫大量小文件）。
# 快路径：直接注入等价环境变量（与 clean-env source 结果一致）。
_apply_conda_humble_fast_env() {
    local prefix="$1"
    export CONDA_PREFIX="${prefix}"
    export PATH="${prefix}/bin${PATH:+:${PATH}}"
    export AMENT_PREFIX_PATH="${prefix}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
    export ROS_ETC_DIR="${prefix}/etc/ros"
    export ROS_DISTRO=humble
    export ROS_VERSION=2
    export ROS_PYTHON_VERSION=3
    # 与 source setup.bash 一致；脚本后面仍可用环境变量覆盖
    export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
    export ROS_OS_OVERRIDE="${ROS_OS_OVERRIDE:-conda:linux}"
    export GSETTINGS_SCHEMA_DIR="${prefix}/share/glib-2.0/schemas${GSETTINGS_SCHEMA_DIR:+:${GSETTINGS_SCHEMA_DIR}}"
    export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    local pp="${prefix}/lib/python311/site-packages:${prefix}/lib/python3.11/site-packages"
    export PYTHONPATH="${pp}${PYTHONPATH:+:${PYTHONPATH}}"
}

set +u
if [[ "${ROS_MODE}" == "conda_humble" && "${FORCE_ROS_SOURCE:-0}" != "1" ]]; then
    _apply_conda_humble_fast_env "${CONDA_PREFIX}"
    echo ">>> ROS fast-env (${ROS_DISTRO} mode=${ROS_MODE}, skip source; FORCE_ROS_SOURCE=1 可强制)"
else
    # setup.bash 依赖相对路径；须在 bash 下 source（勿用 zsh 直接 source）
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
    echo ">>> sourced ${ROS_SETUP} (ROS_DISTRO=${ROS_DISTRO:-?} mode=${ROS_MODE})"
fi
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

if [[ "${ROS_RESTART_DAEMON:-0}" == "1" ]]; then
    ros2 daemon stop >/dev/null 2>&1 || true
    ros2 daemon start >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
# Locale / IME（本机直接用会话 dbus，无需 Docker 的 abstract 代理）
# ---------------------------------------------------------------------------
# 避免每次 locale -a（可能较慢）；需要中文时自行 export LANG
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-${LANG}}"
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
    [[ -f "${src}" ]] || return 0
    if [[ -f "${dst}" ]]; then
        # 已存在则跳过（避免每次 stat/cp）
        return 0
    fi
    mkdir -p "${dst_dir}"
    cp -f "${src}" "${dst}"
    echo ">>> 已复制 fcitx Qt 插件到 ${dst}"
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
    if curl -fsS --max-time 0.3 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
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
    # 不阻塞等待；viewer 侧可稍后探测
}

ensure_remote_qwen_hostctl() {
    local ctl_py="${SCRIPT_DIR}/remote_qwen_hostctl.py"
    local ctl_log="${SCRIPT_DIR}/log/remote_qwen_hostctl.log"
    local ctl_port="${REMOTE_QWEN_CTL_PORT:-18103}"
    mkdir -p "${SCRIPT_DIR}/log"
    if curl -fsS --max-time 0.3 "http://127.0.0.1:${ctl_port}/health" >/dev/null 2>&1; then
        echo ">>> 远程 Qwen hostctl 已在运行 (:${ctl_port})"
        return 0
    fi
    if [[ ! -f "${ctl_py}" ]]; then
        echo "警告: 未找到 ${ctl_py}" >&2
        return 0
    fi
    echo ">>> 启动远程 Qwen hostctl (127.0.0.1:${ctl_port})"
    nohup "${PYTHON}" "${ctl_py}" >>"${ctl_log}" 2>&1 &
    disown || true
}

cleanup_stale_remote_qwen_tunnels() {
    # 隧道已通则跳过；不通也不在启动路径上 pkill/fuser（避免卡顿）
    if curl -fsS --max-time 0.2 "http://127.0.0.1:18100/health" >/dev/null 2>&1 \
        || curl -fsS --max-time 0.2 "http://127.0.0.1:18102/health" >/dev/null 2>&1; then
        echo ">>> 远程 Qwen 隧道已可用"
        return 0
    fi
}

if [[ "${SKIP_HOSTCTL:-0}" != "1" ]]; then
    # 并行探活，减少两次串行 curl 等待
    ensure_local_qwen_hostctl &
    ensure_remote_qwen_hostctl &
    cleanup_stale_remote_qwen_tunnels &
    wait || true
fi

# ---------------------------------------------------------------------------
# Python / Qt：依赖探测有 cache；conda 可走硬编码 Qt 路径
# ---------------------------------------------------------------------------
DEPS_MARKER="${CACHE_DIR}/deps_ok"
_conda_pyqt_root=""
if [[ "${ROS_MODE}" == "conda_humble" && -n "${CONDA_PREFIX:-}" ]]; then
    _conda_pyqt_root="${CONDA_PREFIX}/lib/python3.11/site-packages/PyQt5/Qt5"
fi

_apply_qt_paths() {
    local qt_lib="$1"
    local qt_platforms="$2"
    unset QT_PLUGIN_PATH
    if [[ -n "${qt_lib}" && -d "${qt_lib}" ]]; then
        export LD_LIBRARY_PATH="${qt_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo ">>> PyQt5 Qt lib: ${qt_lib}"
    fi
    if [[ -n "${qt_platforms}" && -d "${qt_platforms}" ]]; then
        export QT_QPA_PLATFORM_PLUGIN_PATH="${qt_platforms}"
    fi
}

_deps_marker_ok() {
    [[ -f "${DEPS_MARKER}" ]] || return 1
    local line
    line="$(head -1 "${DEPS_MARKER}" 2>/dev/null || true)"
    [[ "${line}" == "${PYTHON}|${ROS_MODE}|${CONDA_PREFIX:-}" ]] || return 1
    if [[ -n "${_conda_pyqt_root}" ]]; then
        [[ -d "${_conda_pyqt_root}/lib" && -d "${_conda_pyqt_root}/plugins/platforms" ]] || return 1
    fi
    return 0
}

_write_deps_marker() {
    printf '%s\n' "${PYTHON}|${ROS_MODE}|${CONDA_PREFIX:-}" >"${DEPS_MARKER}"
}

_ensure_fcitx_in_pyqt() {
    local root="$1"
    local src="${EAI_DIR}/qt_plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so"
    [[ -f "${src}" && -n "${root}" ]] || return 0
    local dst_dir="${root}/plugins/platforminputcontexts"
    local dst="${dst_dir}/libfcitxplatforminputcontextplugin.so"
    if [[ -f "${dst}" ]]; then
        return 0
    fi
    mkdir -p "${dst_dir}"
    cp -f "${src}" "${dst}" 2>/dev/null || true
}

skip_probe=0
if [[ "${SKIP_DEPS_CHECK:-0}" == "1" ]]; then
    skip_probe=1
elif [[ "${FORCE_DEPS_CHECK:-0}" != "1" ]] && _deps_marker_ok; then
    skip_probe=1
elif [[ "${FORCE_DEPS_CHECK:-0}" != "1" && "${ROS_MODE}" == "conda_humble" \
    && -n "${_conda_pyqt_root}" \
    && -d "${_conda_pyqt_root}/lib" \
    && -d "${CONDA_PREFIX}/lib/python3.11/site-packages/rclpy" \
    && -d "${CONDA_PREFIX}/lib/python3.11/site-packages/cv_bridge" ]]; then
    # 目录存在即视为就绪，避免每次冷启动 import（share_data 上很慢）
    skip_probe=1
    _write_deps_marker
fi

if [[ "${skip_probe}" -eq 1 && -n "${_conda_pyqt_root}" ]]; then
    _ensure_fcitx_in_pyqt "${_conda_pyqt_root}"
    _apply_qt_paths "${_conda_pyqt_root}/lib" "${_conda_pyqt_root}/plugins/platforms"
elif [[ "${skip_probe}" -eq 1 ]]; then
    echo ">>> 跳过依赖探测（SKIP_DEPS_CHECK / cache）"
else
    _probe="$("${PYTHON}" - <<PY
import pathlib, shutil
status_pip = "ok"
status_ros = "ok"
try:
    import numpy
    import PyQt5  # noqa: F401
    if not str(numpy.__version__).startswith("1."):
        status_pip = "need_pip"
except Exception:
    status_pip = "need_pip"
try:
    import rclpy  # noqa: F401
    from cv_bridge import CvBridge  # noqa: F401
except Exception as exc:
    status_ros = f"fail:{exc}"
print(f"PIP_STATUS={status_pip}")
print(f"ROS_STATUS={status_ros}")
try:
    import PyQt5
    root = pathlib.Path(PyQt5.__file__).resolve().parent / "Qt5"
    src = pathlib.Path(r"${EAI_DIR}/qt_plugins/platforminputcontexts/libfcitxplatforminputcontextplugin.so")
    if src.is_file():
        dst_dir = root / "plugins" / "platforminputcontexts"
        dst = dst_dir / src.name
        if (not dst.exists()) or dst.stat().st_size != src.stat().st_size:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print("FCITX_INSTALLED=1")
    print(f"QT_LIB={root / 'lib'}")
    print(f"QT_PLATFORMS={root / 'plugins' / 'platforms'}")
except Exception:
    pass
PY
)"
    pip_status="$(printf '%s\n' "${_probe}" | sed -n 's/^PIP_STATUS=//p' | tail -1)"
    ros_status="$(printf '%s\n' "${_probe}" | sed -n 's/^ROS_STATUS=//p' | tail -1)"

    need_pip=0
    if [[ "${FORCE_PIP:-0}" == "1" ]]; then
        need_pip=1
    elif [[ "${SKIP_PIP:-0}" != "1" && "${pip_status}" != "ok" ]]; then
        need_pip=1
    fi

    if [[ "${need_pip}" -eq 1 ]]; then
        echo ">>> 安装缺失的 GUI 依赖（pip）…"
        pip_flags=(-q)
        if [[ "${ROS_MODE}" != "conda_humble" ]]; then
            pip_flags+=(--user)
        fi
        pip_index="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
        pip_flags+=(-i "${pip_index}" --trusted-host pypi.tuna.tsinghua.edu.cn)
        "${PYTHON}" -m pip install "${pip_flags[@]}" \
            'PyQt5==5.15.10' \
            'opencv-python-headless==4.10.0.84' \
            'numpy>=1.23.5,<2.0.0' \
            'pyqtgraph==0.13.7' \
            'PyOpenGL==3.1.7' 2>/dev/null || true
        "${PYTHON}" -m pip uninstall -y opencv-python 2>/dev/null || true
        if ! "${PYTHON}" - <<'PY' 2>/dev/null; then
import numpy
import rclpy
from cv_bridge import CvBridge  # noqa: F401
assert numpy.__version__.startswith("1.")
PY
            echo "错误: 依赖未就绪（当前 Python=${PYTHON}）。" >&2
            exit 1
        fi
    elif [[ "${ros_status}" != "ok" ]]; then
        echo "错误: 依赖未就绪（${ros_status:-unknown}；Python=${PYTHON}）。" >&2
        if [[ "${ROS_MODE}" == "conda_humble" ]]; then
            echo "  可重试: FORCE_PIP=1 bash run_local.sh" >&2
        else
            echo "  请先执行: bash install.sh" >&2
        fi
        exit 1
    fi

    if [[ "${_probe}" == *"FCITX_INSTALLED=1"* ]]; then
        echo ">>> 已安装 fcitx Qt 插件"
    fi
    PYQT_QT_LIB="$(printf '%s\n' "${_probe}" | sed -n 's/^QT_LIB=//p' | tail -1)"
    QT5_PLATFORMS="$(printf '%s\n' "${_probe}" | sed -n 's/^QT_PLATFORMS=//p' | tail -1)"
    _apply_qt_paths "${PYQT_QT_LIB}" "${QT5_PLATFORMS}"
    _write_deps_marker
fi
# ---------------------------------------------------------------------------
# 预检（默认跳过；PREFLIGHT=1 启用轻量检查）
# ---------------------------------------------------------------------------
if [[ "${PREFLIGHT:-0}" == "1" && "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
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
    topic_count="${counts%% *}"
    camera_count="${counts##* }"
    echo ">>> 预检: ${topic_count} 个 topic，${camera_count} 个 /camera*"
    if [[ "${PREFLIGHT_HZ:-0}" == "1" ]]; then
        camera_hz="$({ timeout 3 ros2 topic hz /camera/head_color 2>&1 | grep -c "average rate" || true; })"
        if [[ "${topic_count}" -ge 10 && "${camera_hz}" -eq 0 ]]; then
            echo ""
            echo ">>> ⚠️  检测到 topic 但无图像数据 — ROS 很可能在 Docker 内运行" >&2
            echo ">>>     请改用: bash run_in_docker.sh" >&2
            echo ""
        fi
    fi
fi

_log_elapsed
echo ">>> IME QT_IM_MODULE=${QT_IM_MODULE} DBUS=${DBUS_SESSION_BUS_ADDRESS:-<unset>}"
exec "${PYTHON}" "${EAI_DIR}/show_camera_topics.py" "$@"
