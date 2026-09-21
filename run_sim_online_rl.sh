#!/usr/bin/env bash
# 启动 RLinf 仿真在线强化学习（同步 PPO/GRPO 或 Async PPO/SAC）。
# 由 eai viewer「仿真在线强化学习」页调用；也可手动：
#   bash run_sim_online_rl.sh --mode sync --config maniskill_ppo_openvlaoft
#   bash run_sim_online_rl.sh --mode async --config maniskill_async_ppo_openvla
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RLINF_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/RLinf}"
DEFAULT_PY="/home/psibot/miniconda3/envs/RLinf/bin/python"
FALLBACK_PY="/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RLinf/bin/python"
PY="${RLINF_PYTHON:-}"
MODE="sync"
CONFIG_NAME="maniskill_ppo_openvlaoft_quickstart"
ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
EXTRA_OVERRIDES=()
AUTO_SINGLE_GPU="${AUTO_SINGLE_GPU:-1}"

usage() {
  cat <<'EOF'
用法:
  bash run_sim_online_rl.sh [选项] [-- hydra.override=...]

选项:
  --mode sync|async     同步 (run_embodiment.sh) / 异步 (run_async.sh)
  --config NAME         Hydra config 名（examples/embodiment/config/*.yaml）
  --robot-platform P    LIBERO | ALOHA | BRIDGE（默认 LIBERO）
  --rlinf-root PATH     RLinf 仓库路径
  --python PATH         Python 解释器
  --no-auto-single-gpu  禁用单卡自动 placement 覆盖
  -h, --help            显示帮助

说明:
  本机只有 1 张可见 GPU 时，默认自动追加:
    ~cluster.component_placement
    +cluster.component_placement.{actor,env,rollout}=0-0
    以及缩小 env/batch（可用 --no-auto-single-gpu 关闭）。
  单卡推荐 config: maniskill_ppo_openvlaoft_quickstart
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --config)
      CONFIG_NAME="${2:-}"; shift 2 ;;
    --robot-platform)
      ROBOT_PLATFORM="${2:-}"; shift 2 ;;
    --rlinf-root)
      ROOT="${2:-}"; shift 2 ;;
    --python)
      PY="${2:-}"; shift 2 ;;
    --no-auto-single-gpu)
      AUTO_SINGLE_GPU=0; shift ;;
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
if [[ ! -d "${ROOT}/examples/embodiment" ]]; then
  echo "错误: 无效 RLinf 仓库: ${ROOT}" >&2
  exit 1
fi

if [[ -z "${PY}" ]]; then
  if [[ -x "${DEFAULT_PY}" ]]; then
    PY="${DEFAULT_PY}"
  elif [[ -x "${FALLBACK_PY}" ]]; then
    PY="${FALLBACK_PY}"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 RLINF_PYTHON" >&2
    exit 1
  fi
fi
if [[ ! -x "${PY}" ]]; then
  echo "错误: Python 不可执行: ${PY}" >&2
  exit 1
fi

CFG_FILE="${ROOT}/examples/embodiment/config/${CONFIG_NAME}.yaml"
if [[ ! -f "${CFG_FILE}" ]]; then
  echo "错误: 配置不存在: ${CFG_FILE}" >&2
  exit 1
fi

case "${MODE}" in
  sync|embodiment)
    LAUNCH="${ROOT}/examples/embodiment/run_embodiment.sh"
    MODE_LABEL="sync"
    ;;
  async)
    LAUNCH="${ROOT}/examples/embodiment/run_async.sh"
    MODE_LABEL="async"
    ;;
  *)
    echo "错误: --mode 应为 sync 或 async，收到: ${MODE}" >&2
    exit 1
    ;;
esac
if [[ ! -f "${LAUNCH}" ]]; then
  echo "错误: 未找到启动脚本: ${LAUNCH}" >&2
  exit 1
fi

# 让 run_*.sh 里的 `python` 指向指定解释器
PY_BIN_DIR="$(cd "$(dirname "${PY}")" && pwd)"
export PATH="${PY_BIN_DIR}:${PATH}"
export RLINF_ROOT="${ROOT}"
export RLINF_PYTHON="${PY}"
export ROBOT_PLATFORM
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME || true

echo "[sim-online-rl] mode=${MODE_LABEL} config=${CONFIG_NAME} platform=${ROBOT_PLATFORM}"
echo "[sim-online-rl] rlinf=${ROOT}"
echo "[sim-online-rl] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "[sim-online-rl] launch=${LAUNCH}"

# 单卡自动压 placement，避免官方 8 卡 config 断言失败
_has_override_key() {
  local key="$1"
  local item raw
  for item in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
    raw="${item}"
    if [[ "${raw}" == ~* ]]; then
      raw="${raw#\~}"
      [[ "${raw}" == "${key}" || "${raw}" == "${key}."* ]] && return 0
      continue
    fi
    case "${raw}" in
      "${key}="*|"+${key}="*|"~${key}="*|"++${key}="*) return 0 ;;
    esac
  done
  return 1
}

NUM_GPU=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cuda_arr <<< "${CUDA_VISIBLE_DEVICES}"
  for _x in "${_cuda_arr[@]}"; do
    [[ -n "${_x// /}" ]] && NUM_GPU=$((NUM_GPU + 1))
  done
elif command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
fi
echo "[sim-online-rl] visible_gpus=${NUM_GPU}"

if [[ "${AUTO_SINGLE_GPU}" == "1" && "${NUM_GPU}" == "1" ]]; then
  SINGLE_OVERRIDES=(
    "~cluster.component_placement"
    "+cluster.component_placement.actor=0-0"
    "+cluster.component_placement.env=0-0"
    "+cluster.component_placement.rollout=0-0"
    "env.train.total_num_envs=8"
    "env.eval.total_num_envs=4"
    "actor.micro_batch_size=1"
    "actor.global_batch_size=40"
    "actor.enable_offload=True"
    "rollout.enable_offload=True"
    "++env.train.enable_offload=True"
  )
  ADDED=0
  for ov in "${SINGLE_OVERRIDES[@]}"; do
    if [[ "${ov}" == ~* ]]; then
      key="${ov#\~}"
    else
      key="${ov%%=*}"
      key="${key#\+}"
      key="${key#\~}"
    fi
    if ! _has_override_key "${key}"; then
      EXTRA_OVERRIDES+=("${ov}")
      ADDED=$((ADDED + 1))
    fi
  done
  if [[ "${ADDED}" -gt 0 ]]; then
    echo "[sim-online-rl] auto single-GPU overrides (+${ADDED})"
  fi
fi

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  echo "[sim-online-rl] overrides: ${EXTRA_OVERRIDES[*]}"
fi

cd "${ROOT}"
# run_*.sh 只吃 config + platform；额外 hydra override 通过环境变量传给 python 入口较麻烦，
# 这里直接调用 train_*.py（与 shell 脚本等价，但可追加 override）。
EMBODIED_PATH="${ROOT}/examples/embodiment"
export EMBODIED_PATH
export REPO_PATH="${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MODE_LABEL}" == "async" ]]; then
  SRC_FILE="${EMBODIED_PATH}/train_async.py"
else
  SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
fi

LOG_DIR="${ROOT}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
echo "[sim-online-rl] log_dir=${LOG_DIR}"

CMD=(
  python "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"
# 不用 exec：需要 tee 同时写日志文件
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
exit "${PIPESTATUS[0]}"
