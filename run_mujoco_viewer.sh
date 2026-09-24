#!/usr/bin/env bash
# 启动 MuJoCo 交互式图形界面（python -m mujoco.viewer / simulate / EGL 回退）。
# 由 eai viewer「MuJoCo」页调用；也可手动：
#   bash run_mujoco_viewer.sh --mjcf /path/to/model.xml
#   bash run_mujoco_viewer.sh --mjcf model/humanoid/humanoid.xml --mujoco-root /path/to/mujoco
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUJOCO_ROOT_DEFAULT="/share_data/projects/mahjong/share/personal/liyichao/mujoco"
MUJOCO_ROOT="${MUJOCO_ROOT:-$MUJOCO_ROOT_DEFAULT}"
PYTHON_BIN="${MUJOCO_PYTHON:-${PYTHON:-}}"
MJCF=""
MODE="auto" # auto | python | simulate | egl
INSTALL=0
# VirtualGL: use NVIDIA GPU while displaying into TurboVNC.
# Default ON when vglrun exists; set MUJOCO_USE_VGL=0 to force native/Mesa GLX.
VGL_DEVICE="${MUJOCO_VGL_DEVICE:-egl}"

usage() {
  cat <<EOF
Usage:
  bash run_mujoco_viewer.sh [选项]

Options:
  --mjcf PATH           MJCF/XML 模型路径（相对 mujoco-root 或绝对路径）
  --mujoco-root DIR     MuJoCo 源码/模型仓库（默认: ${MUJOCO_ROOT_DEFAULT}）
  --python PATH         带 mujoco 包的 Python
  --mode auto|python|simulate|egl
                        auto: 有 GLX 时用 GLFW viewer（默认经 VirtualGL→NVIDIA），
                              无 GLX 时走 EGL+Tk 回退
  --install             用所选 Python 执行: pip install -U mujoco
  -h, --help

Environment:
  MUJOCO_ROOT / MUJOCO_PYTHON / DISPLAY
  MUJOCO_USE_VGL=0|1    是否用 vglrun（默认: 有 vglrun 则为 1）
  MUJOCO_VGL_DEVICE     传给 vglrun -d（默认 egl → NVIDIA，无需本机 :0）
  MUJOCO_VIEWER_FORCE=1 强制走 GLFW viewer（即使 GLX 实测不可用）

Notes:
  TurboVNC 本身无 GPU GLX，原生会落 Mesa llvmpipe。硬件加速需 VirtualGL：
  vglrun -d egl python -m mujoco.viewer …
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mjcf) MJCF="${2:-}"; shift 2 ;;
    --mujoco-root) MUJOCO_ROOT="${2:-}"; shift 2 ;;
    --python) PYTHON_BIN="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --install) INSTALL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[mujoco] unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

resolve_python() {
  local cand
  if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  for cand in \
    "${MUJOCO_PYTHON:-}" \
    "${PYTHON:-}" \
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/molmospaces/bin/python" \
    "/home/psibot/miniconda3/envs/molmospaces/bin/python" \
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/IsaacLab-Arena/bin/python" \
    "/home/psibot/miniconda3/envs/IsaacLab-Arena/bin/python" \
    "/home/psibot/miniconda3/envs/mujoco/bin/python" \
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/mujoco/bin/python" \
    "/home/psibot/miniconda3/envs/eai/bin/python" \
    "/home/psibot/miniconda3/bin/python" \
    "/usr/bin/python3" \
    "python3"
  do
    [[ -z "${cand}" ]] && continue
    if command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]]; then
      if "${cand}" -c 'import mujoco' >/dev/null 2>&1; then
        echo "${cand}"
        return 0
      fi
    fi
  done
  for cand in \
    "${PYTHON_BIN}" \
    "/home/psibot/miniconda3/bin/python" \
    "/usr/bin/python3" \
    "python3"
  do
    [[ -z "${cand}" ]] && continue
    if command -v "${cand}" >/dev/null 2>&1 || [[ -x "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

find_simulate() {
  local py="$1"
  local cand site
  for cand in \
    "$(command -v simulate 2>/dev/null || true)" \
    "${MUJOCO_ROOT}/build/bin/simulate" \
    "${MUJOCO_ROOT}/build/simulate" \
    "${HOME}/.mujoco/mujoco/bin/simulate"
  do
    [[ -n "${cand}" && -x "${cand}" ]] && { echo "${cand}"; return 0; }
  done
  if [[ -n "${py}" ]]; then
    site="$("${py}" -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)' 2>/dev/null || true)"
    if [[ -n "${site}" ]]; then
      for cand in "${site}/bin/simulate" "${site}/simulate"; do
        [[ -x "${cand}" ]] && { echo "${cand}"; return 0; }
      done
    fi
  fi
  return 1
}

# Usable GLX for GLFW (xdpyinfo listing GLX is not enough on TurboVNC/nodri).
has_glx() {
  local d="${DISPLAY:-}"
  [[ -z "${d}" ]] && return 1
  if command -v glxinfo >/dev/null 2>&1; then
    local out
    out="$(DISPLAY="${d}" glxinfo -B 2>&1 || true)"
    if echo "${out}" | grep -qiE 'Error:|GLX extension not found|couldn.t find RGB GLX'; then
      return 1
    fi
    if echo "${out}" | grep -qiE 'OpenGL renderer string:'; then
      return 0
    fi
    return 1
  fi
  # No glxinfo: unknown → treat as missing so we prefer EGL
  return 1
}

resolve_vglrun() {
  if command -v vglrun >/dev/null 2>&1; then
    echo "vglrun"
  elif [[ -x /opt/VirtualGL/bin/vglrun ]]; then
    echo "/opt/VirtualGL/bin/vglrun"
  fi
}

PY="$(resolve_python || true)"
if [[ -z "${PY}" ]]; then
  echo "[mujoco] 未找到可用 Python" >&2
  exit 1
fi
echo "[mujoco] python=${PY}"
echo "[mujoco] mujoco_root=${MUJOCO_ROOT}"
echo "[mujoco] DISPLAY=${DISPLAY:-}"

if [[ "${INSTALL}" -eq 1 ]]; then
  echo "[mujoco] pip install -U mujoco …"
  "${PY}" -m pip install -U mujoco
  echo "[mujoco] install done"
  if [[ -z "${MJCF}" ]]; then
    exit 0
  fi
fi

if [[ -z "${MJCF}" ]]; then
  echo "[mujoco] 请指定 --mjcf" >&2
  exit 2
fi

# resolve mjcf path
if [[ "${MJCF}" != /* ]]; then
  if [[ -f "${MUJOCO_ROOT}/${MJCF}" ]]; then
    MJCF="${MUJOCO_ROOT}/${MJCF}"
  elif [[ -f "${PWD}/${MJCF}" ]]; then
    MJCF="$(cd "$(dirname "${MJCF}")" && pwd)/$(basename "${MJCF}")"
  fi
fi
MJCF="$(readlink -f "${MJCF}" 2>/dev/null || realpath "${MJCF}" 2>/dev/null || echo "${MJCF}")"
if [[ ! -f "${MJCF}" ]]; then
  echo "[mujoco] 模型不存在: ${MJCF}" >&2
  exit 1
fi
echo "[mujoco] mjcf=${MJCF}"

HAS_PY=0
if "${PY}" -c 'import mujoco' >/dev/null 2>&1; then
  HAS_PY=1
fi
SIM_BIN="$(find_simulate "${PY}" || true)"
VGLRUN="$(resolve_vglrun || true)"
EGL_VIEWER="${SCRIPT_DIR}/tools/mujoco_egl_viewer.py"

GLX_ST=0
has_glx || GLX_ST=$?
if [[ "${GLX_ST}" -eq 0 ]]; then
  echo "[mujoco] GLX=yes (usable OpenGL visual)"
else
  echo "[mujoco] GLX=no/unusable (nodri 或无 RGB GLX visual；GLFW 会失败 → 用 EGL)"
fi

run_egl_viewer() {
  if [[ "${HAS_PY}" -ne 1 ]]; then
    echo "[mujoco] EGL viewer 需要已安装 mujoco 的 Python" >&2
    exit 1
  fi
  if [[ ! -f "${EGL_VIEWER}" ]]; then
    echo "[mujoco] 缺少 ${EGL_VIEWER}" >&2
    exit 1
  fi
  if ! "${PY}" -c 'from PIL import Image, ImageTk; import tkinter' >/dev/null 2>&1; then
    echo "[mujoco] EGL viewer 需要 Pillow + tkinter，请: ${PY} -m pip install pillow" >&2
    exit 1
  fi
  echo "[mujoco] launching EGL viewer: ${PY} ${EGL_VIEWER} --mjcf=${MJCF}"
  cd "${MUJOCO_ROOT}" 2>/dev/null || cd "${SCRIPT_DIR}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  # Qt/cv2 on remote TurboVNC: avoid session-manager auth noise
  unset SESSION_MANAGER || true
  export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
  # Avoid run_local / ROS PYTHONPATH pulling headless cv2 or wrong OpenGL
  export PYTHONPATH=""
  export PYTHONNOUSERSITE=1
  exec "${PY}" -s "${EGL_VIEWER}" --mjcf="${MJCF}"
}

run_python_viewer() {
  # Prefer VirtualGL → NVIDIA. TurboVNC native GLX is Mesa llvmpipe (CPU).
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
  export PYTHONPATH=""
  export PYTHONNOUSERSITE=1
  local cmd=( "${PY}" -s -m mujoco.viewer --mjcf="${MJCF}" )
  local use_vgl="${MUJOCO_USE_VGL:-}"
  if [[ -z "${use_vgl}" ]]; then
    if [[ -n "${VGLRUN}" ]]; then use_vgl=1; else use_vgl=0; fi
  fi

  if [[ "${use_vgl}" == "1" && -n "${VGLRUN}" ]]; then
    echo "[mujoco] launching NVIDIA/VGL: ${VGLRUN} -d ${VGL_DEVICE} ${cmd[*]}  MUJOCO_GL=${MUJOCO_GL}"
    cd "${MUJOCO_ROOT}" 2>/dev/null || cd "${SCRIPT_DIR}"
    if "${VGLRUN}" -d "${VGL_DEVICE}" "${cmd[@]}"; then
      exit 0
    fi
    local rc=$?
    echo "[mujoco] VGL/GLFW 退出 code=${rc}，回退原生 GLFW（可能是 llvmpipe）" >&2
  fi

  echo "[mujoco] launching GLX/GLFW: ${cmd[*]}  MUJOCO_GL=${MUJOCO_GL}"
  cd "${MUJOCO_ROOT}" 2>/dev/null || cd "${SCRIPT_DIR}"
  if "${cmd[@]}"; then
    exit 0
  fi
  local rc=$?
  echo "[mujoco] GLFW viewer 退出 code=${rc}，回退 EGL viewer" >&2
  run_egl_viewer
}

run_simulate() {
  local use_vgl="${MUJOCO_USE_VGL:-}"
  if [[ -z "${use_vgl}" ]]; then
    if [[ -n "${VGLRUN}" ]]; then use_vgl=1; else use_vgl=0; fi
  fi
  if [[ "${use_vgl}" == "1" && -n "${VGLRUN}" ]]; then
    echo "[mujoco] launching NVIDIA/VGL: ${VGLRUN} -d ${VGL_DEVICE} ${SIM_BIN} ${MJCF}"
    cd "$(dirname "${MJCF}")"
    exec "${VGLRUN}" -d "${VGL_DEVICE}" "${SIM_BIN}" "${MJCF}"
  fi
  echo "[mujoco] launching: ${SIM_BIN} ${MJCF}"
  cd "$(dirname "${MJCF}")"
  exec "${SIM_BIN}" "${MJCF}"
}

case "${MODE}" in
  egl)
    run_egl_viewer
    ;;
  python)
    if [[ "${HAS_PY}" -ne 1 ]]; then
      echo "[mujoco] 当前 Python 未安装 mujoco。请点「安装 mujoco」或:" >&2
      echo "  ${PY} -m pip install -U mujoco" >&2
      exit 1
    fi
    if [[ "${GLX_ST}" -ne 0 && "${MUJOCO_VIEWER_FORCE:-0}" != "1" ]]; then
      echo "[mujoco] 无 GLX：GLFW viewer 会失败，改用 EGL 回退（--mode egl）。" >&2
      run_egl_viewer
    fi
    run_python_viewer
    ;;
  simulate)
    if [[ -z "${SIM_BIN}" ]]; then
      echo "[mujoco] 未找到 simulate 可执行文件" >&2
      exit 1
    fi
    if [[ "${GLX_ST}" -ne 0 && "${MUJOCO_VIEWER_FORCE:-0}" != "1" ]]; then
      echo "[mujoco] 无 GLX：simulate 会失败，改用 EGL 回退。" >&2
      run_egl_viewer
    fi
    run_simulate
    ;;
  auto|*)
    if [[ "${HAS_PY}" -eq 1 ]]; then
      if [[ "${GLX_ST}" -eq 0 || "${MUJOCO_VIEWER_FORCE:-0}" == "1" ]]; then
        run_python_viewer
      else
        echo "[mujoco] auto: 无可用 GLX → EGL+Tkinter viewer（非 GLFW）"
        run_egl_viewer
      fi
    elif [[ -n "${SIM_BIN}" ]]; then
      if [[ "${GLX_ST}" -eq 0 || "${MUJOCO_VIEWER_FORCE:-0}" == "1" ]]; then
        run_simulate
      else
        echo "[mujoco] auto: 无 GLX 且无 Python mujoco，无法启动" >&2
        exit 1
      fi
    else
      echo "[mujoco] 未找到 mujoco Python 包或 simulate。" >&2
      echo "  安装: ${PY} -m pip install -U mujoco" >&2
      echo "  或编译: cmake -S ${MUJOCO_ROOT} -B ${MUJOCO_ROOT}/build && cmake --build ${MUJOCO_ROOT}/build" >&2
      exit 1
    fi
    ;;
esac
