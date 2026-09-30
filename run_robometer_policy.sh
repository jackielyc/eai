#!/usr/bin/env bash
# 启动 robometer-policy-learning（训练 / DSRL / 评测）。
# 由 eai viewer「RoboMeter」页调用；也可手动：
#   bash run_robometer_policy.sh --job train --config libero_online_rl
#   bash run_robometer_policy.sh --job eval_trained_dsrl --config eval_trained_dsrl --gui
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="/share_data/projects/mahjong/share/personal/liyichao/robometer-policy-learning"
ROOT="${ROBOMETER_ROOT:-${DEFAULT_ROOT}}"
PY="${ROBOMETER_PYTHON:-}"
JOB="train"
CONFIG_NAME=""
GUI=0
EXTRA_OVERRIDES=()

usage() {
  cat <<'EOF'
用法:
  bash run_robometer_policy.sh [选项] [-- hydra.override=...]

选项:
  --job NAME            train | train_dsrl | eval_pi0 | eval_trained_dsrl |
                        train_async | relabel_server
  --config NAME         Hydra config（robometer_policy_learning/configs/*.yaml）
  --root PATH           robometer-policy-learning 仓库根
  --python PATH         Python 解释器（建议仓库 .venv 或 robometer-policy conda）
  --gui                 评测等任务弹出 MuJoCo 窗口（glfw）；train 固定 egl 离屏
  -h, --help            显示帮助

工作目录为仓库根。入口见 scripts/train.py、scripts/train_dsrl.py、
scripts/eval_pi0.py、scripts/eval_trained_dsrl.py。
默认 Python：miniconda3/envs/robometer-policy → 仓库 .venv。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB="${2:-}"; shift 2 ;;
    --config)
      CONFIG_NAME="${2:-}"; shift 2 ;;
    --root)
      ROOT="${2:-}"; shift 2 ;;
    --python)
      PY="${2:-}"; shift 2 ;;
    --gui)
      GUI=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      EXTRA_OVERRIDES+=("$@")
      break ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift ;;
  esac
done

ROOT="$(cd "${ROOT}" && pwd)"
if [[ ! -d "${ROOT}/robometer_policy_learning" ]]; then
  echo "错误: 无效仓库（缺少 robometer_policy_learning/）: ${ROOT}" >&2
  exit 1
fi

JOB_NORM="$(echo "${JOB}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
case "${JOB_NORM}" in
  train|online_rl) JOB_NORM="train"; SCRIPT_REL="scripts/train.py"; DEFAULT_CFG="libero_online_rl" ;;
  train_dsrl|dsrl) JOB_NORM="train_dsrl"; SCRIPT_REL="scripts/train_dsrl.py"; DEFAULT_CFG="dsrl_libero_config" ;;
  eval_pi0|pi0) JOB_NORM="eval_pi0"; SCRIPT_REL="scripts/eval_pi0.py"; DEFAULT_CFG="eval_pi0" ;;
  eval_trained_dsrl|eval_dsrl|eval) JOB_NORM="eval_trained_dsrl"; SCRIPT_REL="scripts/eval_trained_dsrl.py"; DEFAULT_CFG="eval_trained_dsrl" ;;
  train_async|async) JOB_NORM="train_async"; SCRIPT_REL="scripts/train_async.py"; DEFAULT_CFG="config_distributed" ;;
  relabel_server|relabel) JOB_NORM="relabel_server"; SCRIPT_REL="scripts/start_reward_relabel_server.py"; DEFAULT_CFG="reward_relabel_server" ;;
  *)
    echo "错误: 未知 --job: ${JOB}" >&2
    exit 1
    ;;
esac

CONFIG_NAME="${CONFIG_NAME:-${DEFAULT_CFG}}"
CFG_FILE="${ROOT}/robometer_policy_learning/configs/${CONFIG_NAME}.yaml"
if [[ ! -f "${CFG_FILE}" ]]; then
  echo "错误: 配置不存在: ${CFG_FILE}" >&2
  exit 1
fi

SCRIPT_PATH="${ROOT}/${SCRIPT_REL}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "错误: 未找到脚本: ${SCRIPT_PATH}" >&2
  exit 1
fi

if [[ -z "${PY}" ]]; then
  # 优先独立 conda 环境（完整依赖）；仓库 .venv 可能只是 uv 空壳且看不到 conda site-packages
  if [[ -x "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/robometer-policy/bin/python" ]]; then
    PY="/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/robometer-policy/bin/python"
  elif [[ -x "/home/psibot/miniconda3/envs/robometer-policy/bin/python" ]]; then
    PY="/home/psibot/miniconda3/envs/robometer-policy/bin/python"
  elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PY="${ROOT}/.venv/bin/python"
  elif [[ -x "${ROOT}/venv/bin/python" ]]; then
    PY="${ROOT}/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 ROBOMETER_PYTHON 或 --python" >&2
    exit 1
  fi
fi
if [[ ! -x "${PY}" ]]; then
  echo "错误: Python 不可执行: ${PY}" >&2
  exit 1
fi

PY_BIN_DIR="$(cd "$(dirname "${PY}")" && pwd)"
export PATH="${PY_BIN_DIR}:${PATH}"
export ROBOMETER_ROOT="${ROOT}"
export ROBOMETER_PYTHON="${PY}"
# 未 pip install -e 本仓库时，把仓库根加入 path
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
unset PYTHONHOME || true

# NVIDIA pip wheels: ensure cudnn/cublas are findable (avoids
# "Invalid handle. Cannot load symbol cudnnGetVersion" when parent Qt/GL
# pollutes the process environment).
_SP="$("${PY}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -n "${_SP}" && -d "${_SP}/nvidia" ]]; then
  _NV_LIBS=()
  while IFS= read -r _d; do
    [[ -n "${_d}" ]] && _NV_LIBS+=("${_d}")
  done < <(find "${_SP}/nvidia" -type d -name lib 2>/dev/null | sort)
  if [[ ${#_NV_LIBS[@]} -gt 0 ]]; then
    _NV_JOIN="$(IFS=:; echo "${_NV_LIBS[*]}")"
    export LD_LIBRARY_PATH="${_NV_JOIN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi
# Drop Qt/OpenGL leftovers inherited from eai viewer QProcess.
unset QT_XCB_GL_INTEGRATION QT_OPENGL QT_QUICK_BACKEND __GL_SYNC_TO_VBLANK || true

# LIBERO train uses OffScreenRenderEnv (no on-screen MuJoCo window). glfw under a
# Qt parent often breaks cudnn; keep egl for train/train_async even if --gui.
if [[ "${GUI}" == "1" && "${JOB_NORM}" != "train" && "${JOB_NORM}" != "train_async" ]]; then
  export MUJOCO_GL=glfw
  unset PYOPENGL_PLATFORM || true
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "[warn] DISPLAY 为空，图形窗口可能无法弹出（本机桌面/X11 转发）" >&2
  fi
else
  # Must overwrite (not :- default): GUI/QProcess may inherit MUJOCO_GL=glfw.
  if [[ "${GUI}" == "1" ]]; then
    echo "[robometer] train 使用 OffScreenRenderEnv，忽略 --gui，强制 MUJOCO_GL=egl" >&2
  fi
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
fi

# LIBERO first-import writes ~/.libero/config.yaml via interactive input; pre-seed it.
LIBERO_CFG_DIR="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
LIBERO_CFG_FILE="${LIBERO_CFG_DIR}/config.yaml"
if [[ ! -f "${LIBERO_CFG_FILE}" ]]; then
  mkdir -p "${LIBERO_CFG_DIR}"
  LIBERO_PKG=""
  for cand in \
    "${ROOT}/LIBERO/libero/libero" \
    "${ROOT}/libero/libero"; do
    if [[ -d "${cand}" ]]; then
      LIBERO_PKG="$(cd "${cand}" && pwd)"
      break
    fi
  done
  if [[ -n "${LIBERO_PKG}" ]]; then
    cat >"${LIBERO_CFG_FILE}" <<EOF
assets: ${LIBERO_PKG}/assets
bddl_files: ${LIBERO_PKG}/bddl_files
benchmark_root: ${LIBERO_PKG}
datasets: $(cd "${LIBERO_PKG}/.." && pwd)/datasets
init_states: ${LIBERO_PKG}/init_files
EOF
    echo "[robometer] seeded LIBERO config: ${LIBERO_CFG_FILE}"
  else
    echo "[warn] LIBERO package path not found; first import may prompt for dataset path" >&2
  fi
fi

cd "${ROOT}"
LOG_DIR="${ROOT}/outputs"
mkdir -p "${LOG_DIR}"
echo "[robometer] log_dir=${LOG_DIR}"
echo "[robometer] job=${JOB_NORM} config=${CONFIG_NAME} gui=${GUI}"
echo "[robometer] python=${PY}"
echo "[robometer] cwd=${ROOT}"

CMD=("${PY}" "${SCRIPT_PATH}" "--config-name=${CONFIG_NAME}")
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "$ ${CMD[*]}"
exec "${CMD[@]}"
