#!/usr/bin/env bash
# 在 RoboDojo / RISE 中跑基于 RLinf 的在线强化学习（policy_online）。
# 由 eai viewer「RoboDojo在线RL」页调用；也可手动：
#   bash run_robodojo_online_rl.sh --config rl_release
#   bash run_robodojo_online_rl.sh --mode multi --config rl_release
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RISE_DEFAULT="/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/XPolicyLab/policy/RISE/RISE"
RISE_ROOT="${RISE_ROOT:-${RISE_DEFAULT}}"
DEFAULT_PY="/home/psibot/miniconda3/envs/RLinf/bin/python"
FALLBACK_PY="/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RLinf/bin/python"
ROBODOJO_PY="/share_data/projects/mahjong/share/personal/liyichao/RoboDojo_cache/envs/RoboDojo/bin/python"
PY="${RISE_PYTHON:-${RLINF_PYTHON:-}}"
MODE="single"
CONFIG_NAME="rl_release"
EXTRA_OVERRIDES=()
AUTO_SINGLE_GPU="${AUTO_SINGLE_GPU:-1}"

usage() {
  cat <<'EOF'
用法:
  bash run_robodojo_online_rl.sh [选项] [-- hydra.override=...]

选项:
  --mode single|multi   单机 run_embodiment.sh / 多机 Ray unified
  --config NAME         Hydra config（examples/embodiment/config/*.yaml）
  --rise-root PATH      RISE 仓库根（含 policy_and_value/）
  --python PATH         Python 解释器（建议 rise / RLinf env）
  --no-auto-single-gpu  禁用单卡自动 placement 覆盖
  -h, --help            显示帮助

说明:
  工作目录为 RISE 根。官方入口见
  XPolicyLab/policy/RISE/RISE/docs/online_training.md
  单卡时默认追加 env/rollout/actor=0-0 并缩小 batch。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --config)
      CONFIG_NAME="${2:-}"; shift 2 ;;
    --rise-root)
      RISE_ROOT="${2:-}"; shift 2 ;;
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

RISE_ROOT="$(cd "${RISE_ROOT}" && pwd)"
ONLINE_ROOT="${RISE_ROOT}/policy_and_value/policy_online"
EMBODIED_PATH="${ONLINE_ROOT}/examples/embodiment"
if [[ ! -d "${EMBODIED_PATH}" ]]; then
  echo "错误: 无效 RISE 仓库（缺少 policy_online/examples/embodiment）: ${RISE_ROOT}" >&2
  exit 1
fi

if [[ -z "${PY}" ]]; then
  if [[ -x "${DEFAULT_PY}" ]]; then
    PY="${DEFAULT_PY}"
  elif [[ -x "${FALLBACK_PY}" ]]; then
    PY="${FALLBACK_PY}"
  elif [[ -x "${ROBODOJO_PY}" ]]; then
    PY="${ROBODOJO_PY}"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 RISE_PYTHON 或 RLINF_PYTHON" >&2
    exit 1
  fi
fi
if [[ ! -x "${PY}" ]]; then
  echo "错误: Python 不可执行: ${PY}" >&2
  exit 1
fi

CFG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
if [[ ! -f "${CFG_FILE}" ]]; then
  echo "错误: 配置不存在: ${CFG_FILE}" >&2
  exit 1
fi

MODE_NORM="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')"
case "${MODE_NORM}" in
  single|sync|embodiment) MODE_NORM="single" ;;
  multi|ray|multinode) MODE_NORM="multi" ;;
  *)
    echo "错误: --mode 应为 single 或 multi，收到: ${MODE}" >&2
    exit 1
    ;;
esac

PY_BIN_DIR="$(cd "$(dirname "${PY}")" && pwd)"
export PATH="${PY_BIN_DIR}:${PATH}"
export RISE_ROOT
export RISE_PYTHON="${PY}"
export EMBODIED_PATH
export REPO_PATH="${ONLINE_ROOT}"
# openpi_value 在 offline 包的 src/；dynamics_model 在 dynamics/
OPENPI_VALUE_SRC="${RISE_ROOT}/policy_and_value/policy_offline_and_value/src"
DYNAMICS_ROOT="${RISE_ROOT}/dynamics"
if [[ ! -d "${OPENPI_VALUE_SRC}/openpi_value" ]]; then
  echo "错误: 未找到 openpi_value: ${OPENPI_VALUE_SRC}/openpi_value" >&2
  exit 1
fi
if [[ ! -d "${DYNAMICS_ROOT}/dynamics_model" ]]; then
  echo "错误: 未找到 dynamics_model: ${DYNAMICS_ROOT}/dynamics_model" >&2
  exit 1
fi
export OPENPI_VALUE_SRC DYNAMICS_ROOT
export PYTHONPATH="${ONLINE_ROOT}:${OPENPI_VALUE_SRC}:${DYNAMICS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Ray dashboard 默认 8888 常被占用，换端口避免 ERROR 刷屏
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,graphics}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME || true

# 动力学模型权重（LTX backbone + RISE pretrained）；infer.yaml 里 path/checkpoints 是占位符
RISE_DYNAMICS_ROOT="${RISE_DYNAMICS_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/models/rise_dynamics}"
RISE_DYNAMICS_INFER="${RISE_DYNAMICS_INFER:-${RISE_DYNAMICS_ROOT}/infer.yaml}"
RISE_DYNAMICS_BACKBONE="${RISE_DYNAMICS_BACKBONE:-${RISE_DYNAMICS_ROOT}/checkpoints}"
RISE_DYNAMICS_WEIGHT="${RISE_DYNAMICS_WEIGHT:-${RISE_DYNAMICS_ROOT}/pretrained/diffusion_pytorch_model.safetensors}"

_require_dynamics_assets() {
  local ok=1
  if [[ ! -f "${RISE_DYNAMICS_BACKBONE}/tokenizer/tokenizer_config.json" ]]; then
    echo "错误: 缺少 LTX tokenizer: ${RISE_DYNAMICS_BACKBONE}/tokenizer/" >&2
    ok=0
  fi
  if [[ ! -f "${RISE_DYNAMICS_BACKBONE}/text_encoder/config.json" ]]; then
    echo "错误: 缺少 LTX text_encoder: ${RISE_DYNAMICS_BACKBONE}/text_encoder/" >&2
    ok=0
  fi
  if [[ ! -f "${RISE_DYNAMICS_BACKBONE}/vae/config.json" ]]; then
    echo "错误: 缺少 LTX vae: ${RISE_DYNAMICS_BACKBONE}/vae/" >&2
    ok=0
  fi
  if [[ ! -f "${RISE_DYNAMICS_WEIGHT}" ]]; then
    echo "错误: 缺少动力学权重: ${RISE_DYNAMICS_WEIGHT}" >&2
    ok=0
  fi
  if [[ ! -f "${RISE_DYNAMICS_INFER}" ]]; then
    echo "错误: 缺少动力学 infer 配置: ${RISE_DYNAMICS_INFER}" >&2
    ok=0
  fi
  if [[ "${ok}" -ne 1 ]]; then
    echo "请先下载（约需数 GB）：" >&2
    echo "  bash ${SCRIPT_DIR}/scripts/setup_rise_dynamics_ckpts.sh" >&2
    echo "或参考 RISE/docs/dynamics_model.md（download.sh + OpenDriveLab RISE_Assets）。" >&2
    exit 1
  fi
}

cd "${RISE_ROOT}"

echo "[robodojo-online-rl] mode=${MODE_NORM} config=${CONFIG_NAME}"
echo "[robodojo-online-rl] rise=${RISE_ROOT}"
echo "[robodojo-online-rl] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"
echo "[robodojo-online-rl] PYTHONPATH openpi_value=$(python -c 'import openpi_value,os; print(os.path.dirname(openpi_value.__file__))')"
echo "[robodojo-online-rl] PYTHONPATH dynamics_model=$(python -c 'import dynamics_model,os; print(os.path.dirname(dynamics_model.__file__))')"

# 启动前快速检查 openpi_value 关键依赖（缺失时给出安装提示）
if ! python -c "import tqdm_loggable, flax, openpi_value.shared.download" >/dev/null 2>&1; then
  echo "错误: RLinf 环境缺少 openpi_value 依赖（如 tqdm_loggable / flax）。" >&2
  echo "请先运行: bash ${SCRIPT_DIR}/scripts/setup_robodojo_online_rl_deps.sh ${PY}" >&2
  exit 1
fi

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

# rl_release 默认 add_dynamics_model=true；若用户未覆盖 config 则校验并注入本地 infer.yaml
NEED_DYNAMICS=1
if _has_override_key "actor.model.openpi.add_dynamics_model"; then
  for item in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
    case "${item}" in
      actor.model.openpi.add_dynamics_model=False|actor.model.openpi.add_dynamics_model=false|actor.model.openpi.add_dynamics_model=0)
        NEED_DYNAMICS=0 ;;
    esac
  done
fi
if [[ "${NEED_DYNAMICS}" == "1" ]]; then
  _require_dynamics_assets
  if ! _has_override_key "actor.model.openpi.dynamics_model_config"; then
    EXTRA_OVERRIDES+=("actor.model.openpi.dynamics_model_config=${RISE_DYNAMICS_INFER}")
    echo "[robodojo-online-rl] dynamics_model_config=${RISE_DYNAMICS_INFER}"
  fi
fi

NUM_GPU=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _cuda_arr <<< "${CUDA_VISIBLE_DEVICES}"
  for _x in "${_cuda_arr[@]}"; do
    [[ -n "${_x// /}" ]] && NUM_GPU=$((NUM_GPU + 1))
  done
elif command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
fi
echo "[robodojo-online-rl] visible_gpus=${NUM_GPU}"

if [[ "${AUTO_SINGLE_GPU}" == "1" && "${NUM_GPU}" == "1" ]]; then
  SINGLE_OVERRIDES=(
    "~cluster.component_placement"
    "+cluster.component_placement.env=0-0"
    "+cluster.component_placement.rollout=0-0"
    "+cluster.component_placement.actor=0-0"
    "algorithm.num_group_envs=4"
    "actor.micro_batch_size=1"
    "actor.global_batch_size=4"
    "actor.enable_offload=True"
    "env.enable_offload=True"
    "rollout.enable_offload=True"
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
    echo "[robodojo-online-rl] auto single-GPU overrides (+${ADDED})"
  fi
fi

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  echo "[robodojo-online-rl] overrides: ${EXTRA_OVERRIDES[*]}"
fi

# 与官方 run_embodiment.sh 一致：必要时覆盖 transformers optimization
OPT_SRC="${ONLINE_ROOT}/rlinf/module2replace/optimization.py"
if [[ -f "${OPT_SRC}" ]]; then
  TRANSFORMERS_DIR="$(python -c 'import os,transformers; print(os.path.dirname(transformers.__file__))' 2>/dev/null || true)"
  if [[ -n "${TRANSFORMERS_DIR}" && -d "${TRANSFORMERS_DIR}" ]]; then
    cp "${OPT_SRC}" "${TRANSFORMERS_DIR}/optimization.py" || true
  fi
fi

LOG_DIR="${ONLINE_ROOT}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
echo "[robodojo-online-rl] log_dir=${LOG_DIR}"

if [[ "${MODE_NORM}" == "multi" ]]; then
  # 多机：先起 Ray，再在 rank0 上训；仍把 hydra override 交给 train 入口
  RAY_SCRIPT="${ONLINE_ROOT}/ray_utils/start_ray_unified_multi_task.sh"
  if [[ -f "${RAY_SCRIPT}" ]]; then
    echo "[robodojo-online-rl] starting ray via ${RAY_SCRIPT}"
    # shellcheck disable=SC1090
    bash "${RAY_SCRIPT}" || true
  fi
fi

SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
CMD=(
  python "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
)
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
echo >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
exit "${PIPESTATUS[0]}"
