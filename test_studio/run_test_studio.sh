#!/usr/bin/env bash
# 独立启动「测试工作室」前端（不打开相机 Viewer，不是 Tab）
#
# 用法:
#   bash test_studio/run_test_studio.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAI_DIR_HOST="$(readlink -f "${SCRIPT_DIR}/..")"
CONTAINER="${A2D_DOCKER_CONTAINER:-a2d-tele-release-2-1-0rc3-latest}"
PSIBOT_HOME="${PSIBOT_HOME:-/home/psibot}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DISPLAY="${DISPLAY:-:0}"
IME_DBUS_ABSTRACT="${IME_DBUS_ABSTRACT:-a2d_ime_dbus}"
DBUS_PROXY_ADDR="unix:abstract=${IME_DBUS_ABSTRACT}"
REMOTE_DIR="${EAI_DIR_HOST}"
WORKSPACE_INSTALL="${PSIBOT_HOME}/workspace_liyichao/install/setup.bash"
A2D_SCRIPTS_DIR="${A2D_SCRIPTS_DIR:-${PSIBOT_HOME}/workspace_liyichao/install/scripts_pack/share/scripts_pack/scripts}"
A2D_SDK_HOME="${A2D_SDK_HOME:-${PSIBOT_HOME}/a2d_sdk}"
FASTDDS_XML="${A2D_SDK_HOME}/dds/fastdds_profiles_a2d.xml"

if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "错误: 容器 ${CONTAINER} 不存在，请先启动 a2d 容器" >&2
  exit 1
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local: >/dev/null 2>&1 || true
fi

XAUTH_COPY="${PSIBOT_HOME}/.cache/a2d_xauth"
mkdir -p "${PSIBOT_HOME}/.cache" "${EAI_DIR_HOST}/log"
if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
  cp -f "${XAUTHORITY}" "${XAUTH_COPY}" 2>/dev/null || true
elif [[ -f "${HOME}/.Xauthority" ]]; then
  cp -f "${HOME}/.Xauthority" "${XAUTH_COPY}" 2>/dev/null || true
fi
chmod 644 "${XAUTH_COPY}" 2>/dev/null || true

# 推理 hostctl（与 viewer 相同）
if ! curl -fsS --max-time 1 "http://127.0.0.1:${LOCAL_QWEN_CTL_PORT:-18101}/health" >/dev/null 2>&1; then
  nohup python3 "${EAI_DIR_HOST}/local_qwen_hostctl.py" >>"${EAI_DIR_HOST}/log/local_qwen_hostctl.log" 2>&1 &
  disown || true
fi
if ! curl -fsS --max-time 1 "http://127.0.0.1:${REMOTE_QWEN_CTL_PORT:-18103}/health" >/dev/null 2>&1; then
  nohup python3 "${EAI_DIR_HOST}/remote_qwen_hostctl.py" >>"${EAI_DIR_HOST}/log/remote_qwen_hostctl.log" 2>&1 &
  disown || true
fi

# 尽量复用 run_in_docker 已准备好的 fcitx abstract dbus 代理；若无则提示
if ! docker exec -u "${HOST_UID}:${HOST_GID}" "${CONTAINER}" \
  bash -lc "test -n \"\${DBUS_SESSION_BUS_ADDRESS:-}\"" 2>/dev/null; then
  :
fi

FASTDDS_IN_CONTAINER="${FASTDDS_XML}"
if ! docker exec -u "${HOST_UID}:${HOST_GID}" "${CONTAINER}" test -f "${FASTDDS_XML}" 2>/dev/null; then
  docker exec -u "${HOST_UID}:${HOST_GID}" "${CONTAINER}" mkdir -p /tmp/a2d_dds 2>/dev/null || true
  if [[ -f "${FASTDDS_XML}" ]]; then
    docker cp "${FASTDDS_XML}" "${CONTAINER}:/tmp/a2d_dds/fastdds_profiles_a2d.xml" 2>/dev/null || true
  fi
  FASTDDS_IN_CONTAINER="/tmp/a2d_dds/fastdds_profiles_a2d.xml"
fi

echo ">>> 独立启动测试工作室（非 Viewer Tab）"
echo ">>> 容器: ${CONTAINER}"
echo ">>> 目录: ${REMOTE_DIR}"

exec docker exec \
  -u "${HOST_UID}:${HOST_GID}" \
  -e HOME="${PSIBOT_HOME}" \
  -e USER="${USER:-psibot}" \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -e "XDG_RUNTIME_DIR=/tmp/a2d_runtime" \
  -e "DBUS_SESSION_BUS_ADDRESS=${DBUS_PROXY_ADDR}" \
  -e "LANG=zh_CN.UTF-8" \
  -e "LC_ALL=zh_CN.UTF-8" \
  -e "QT_IM_MODULE=fcitx" \
  -e "GTK_IM_MODULE=fcitx" \
  -e "XMODIFIERS=@im=fcitx" \
  -e "XAUTHORITY=${XAUTH_COPY}" \
  -e "FASTRTPS_DEFAULT_PROFILES_FILE=${FASTDDS_IN_CONTAINER}" \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e ROS_DOMAIN_ID=0 \
  -e ROS_LOCALHOST_ONLY=1 \
  -e "A2D_SCRIPTS_DIR=${A2D_SCRIPTS_DIR}" \
  -e "PSIBOT_HOME=${PSIBOT_HOME}" \
  -e "LOCAL_QWEN_CTL_URL=http://127.0.0.1:${LOCAL_QWEN_CTL_PORT:-18101}" \
  -e "LOCAL_QWEN_API_BASE=http://127.0.0.1:${LOCAL_QWEN_PORT:-8100}/v1" \
  -e "REMOTE_QWEN_CTL_URL=http://127.0.0.1:${REMOTE_QWEN_CTL_PORT:-18103}" \
  -it "${CONTAINER}" \
  bash -lc "
set -eo pipefail
mkdir -p /tmp/a2d_runtime ${PSIBOT_HOME}/.cache/a2d_runtime
chmod 700 /tmp/a2d_runtime ${PSIBOT_HOME}/.cache/a2d_runtime 2>/dev/null || true
source /opt/ros/humble/setup.bash
if [[ -f /opt/psi/rt/a2d-tele/install/setup.bash ]]; then
  source /opt/psi/rt/a2d-tele/install/setup.bash
fi
if [[ -f ${WORKSPACE_INSTALL} ]]; then
  set +u
  source ${WORKSPACE_INSTALL}
  set -e
fi
export HOME=\"${PSIBOT_HOME}\"
export QT_IM_MODULE=fcitx XMODIFIERS=@im=fcitx GTK_IM_MODULE=fcitx
export DBUS_SESSION_BUS_ADDRESS=\"${DBUS_PROXY_ADDR}\"
export XDG_RUNTIME_DIR=/tmp/a2d_runtime
unset QT_PLUGIN_PATH
python3 -m pip install -q --user \
  'PyQt5==5.15.10' 'opencv-python-headless==4.10.0.84' \
  'numpy>=1.23.5,<2.0.0' 'pyqtgraph==0.13.7' 'PyOpenGL==3.1.7' 2>/dev/null || true
python3 -m pip uninstall -y opencv-python 2>/dev/null || true
PYQT_QT_LIB=\"\$(python3 -c 'import PyQt5, pathlib; print(pathlib.Path(PyQt5.__file__).resolve().parent / \"Qt5\" / \"lib\")' 2>/dev/null || true)\"
if [[ -n \"\${PYQT_QT_LIB}\" && -d \"\${PYQT_QT_LIB}\" ]]; then
  export LD_LIBRARY_PATH=\"\${PYQT_QT_LIB}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}\"
fi
cd \"${REMOTE_DIR}\"
export PYTHONPATH=\"${REMOTE_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}\"
echo \">>> python3 -m test_studio.main\"
exec python3 -m test_studio.main
"
