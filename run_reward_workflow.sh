#!/usr/bin/env bash
# 启动 RLinf 奖励模型工作流：数据采集 / 预处理 / 训练 / 预处理→训练。
# 由 eai viewer「Reward训练」页调用；也可手动：
#   bash run_reward_workflow.sh --mode preprocess --raw-data-path /path/to/collected_data
#   bash run_reward_workflow.sh --mode train --config reward_training \
#       data.train_data_paths=/path/train.pt data.val_data_paths=/path/val.pt
#   bash run_reward_workflow.sh --mode pipeline --raw-data-path /path/to/collected_data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RLINF_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/RLinf}"
DEFAULT_PY="/home/psibot/miniconda3/envs/RLinf/bin/python"
FALLBACK_PY="/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RLinf/bin/python"
PY="${RLINF_PYTHON:-}"
MODE="preprocess"
CONFIG_NAME="reward_training"
DATASET_TYPE="resnet"
RAW_DATA_PATH=""
OUTPUT_DIR=""
NUM_SAMPLES_PER_EPISODE=0
VAL_SPLIT="0.2"
FAIL_SUCCESS_RATIO="2.0"
SEED=42
KEEP_LAST_FRAME=1
EXTRA_OVERRIDES=()

usage() {
  cat <<'EOF'
用法:
  bash run_reward_workflow.sh [选项] [-- hydra.override=...]

选项:
  --mode MODE           preprocess | train | collect | pipeline
  --config NAME         Hydra config（train/collect；默认 reward_training /
                        realworld_collect_dataset）
  --dataset-type TYPE   resnet | qwentrend（预处理脚本；默认 resnet）
  --raw-data-path PATH  原始 episode .pkl 目录（preprocess / pipeline）
  --output-dir PATH     预处理输出目录（默认 logs/processed_reward_data）
  --num-samples N       每 episode 采样帧数（0=全部）
  --val-split F         验证集比例（默认 0.2）
  --fail-success-ratio R  fail:success 采样比（默认 2.0）
  --seed N              随机种子（默认 42）
  --no-keep-last-frame  采样时不强制保留最后一帧
  --rlinf-root PATH     RLinf 仓库路径
  --python PATH         Python 解释器
  -h, --help            显示帮助

说明:
  pipeline = preprocess → train，并自动把 data.train/val_data_paths 指到
  output-dir 下的 train.pt / val.pt（qwentrend 则指 train.jsonl / val.jsonl）。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --config)
      CONFIG_NAME="${2:-}"; shift 2 ;;
    --dataset-type)
      DATASET_TYPE="${2:-}"; shift 2 ;;
    --raw-data-path)
      RAW_DATA_PATH="${2:-}"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2 ;;
    --num-samples)
      NUM_SAMPLES_PER_EPISODE="${2:-}"; shift 2 ;;
    --val-split)
      VAL_SPLIT="${2:-}"; shift 2 ;;
    --fail-success-ratio)
      FAIL_SUCCESS_RATIO="${2:-}"; shift 2 ;;
    --seed)
      SEED="${2:-}"; shift 2 ;;
    --no-keep-last-frame)
      KEEP_LAST_FRAME=0; shift ;;
    --rlinf-root)
      ROOT="${2:-}"; shift 2 ;;
    --python)
      PY="${2:-}"; shift 2 ;;
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
REWARD_DIR="${ROOT}/examples/reward"
if [[ ! -d "${REWARD_DIR}" ]]; then
  echo "错误: 无效 RLinf 仓库（缺少 examples/reward）: ${ROOT}" >&2
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

MODE_NORM="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]')"
DATASET_TYPE="$(echo "${DATASET_TYPE}" | tr '[:upper:]' '[:lower:]')"
case "${DATASET_TYPE}" in
  resnet|qwentrend) ;;
  *)
    echo "错误: --dataset-type 应为 resnet 或 qwentrend，收到: ${DATASET_TYPE}" >&2
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ "${DATASET_TYPE}" == "qwentrend" ]]; then
    OUTPUT_DIR="${ROOT}/logs/processed_qwentrend_reward_data"
  else
    OUTPUT_DIR="${ROOT}/logs/processed_reward_data"
  fi
fi
# 相对路径相对 RLinf 根
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ROOT}/${OUTPUT_DIR}"
fi

PY_BIN_DIR="$(cd "$(dirname "${PY}")" && pwd)"
export PATH="${PY_BIN_DIR}:${PATH}"
export RLINF_ROOT="${ROOT}"
export RLINF_PYTHON="${PY}"
export REWARD_PATH="${REWARD_DIR}"
export EMBODIED_PATH="${REWARD_DIR}"
export REPO_PATH="${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HYDRA_FULL_ERROR=1
unset PYTHONHOME || true

cd "${ROOT}"

echo "[reward-workflow] mode=${MODE_NORM} dataset_type=${DATASET_TYPE}"
echo "[reward-workflow] rlinf=${ROOT}"
echo "[reward-workflow] python=$(command -v python) ($(python -c 'import sys; print(sys.version.split()[0])'))"

run_preprocess() {
  local raw="$1"
  local out="$2"
  if [[ -z "${raw}" ]]; then
    echo "错误: preprocess/pipeline 需要 --raw-data-path" >&2
    exit 1
  fi
  if [[ ! -d "${raw}" ]]; then
    echo "错误: raw-data-path 不是目录: ${raw}" >&2
    exit 1
  fi
  mkdir -p "${out}"
  local src
  local args
  if [[ "${DATASET_TYPE}" == "qwentrend" ]]; then
    src="${REWARD_DIR}/preprocess_qwentrend_reward_dataset.py"
    args=(
      "${src}"
      --raw-data-path "${raw}"
      --output-dir "${out}"
      --num-samples-per-episode "${NUM_SAMPLES_PER_EPISODE}"
      --val-split "${VAL_SPLIT}"
      --seed "${SEED}"
    )
    if [[ "${KEEP_LAST_FRAME}" == "0" ]]; then
      args+=(--no-keep-last-window)
    fi
  else
    src="${REWARD_DIR}/preprocess_reward_dataset.py"
    args=(
      "${src}"
      --raw-data-path "${raw}"
      --output-dir "${out}"
      --num-samples-per-episode "${NUM_SAMPLES_PER_EPISODE}"
      --val-split "${VAL_SPLIT}"
      --fail-success-ratio "${FAIL_SUCCESS_RATIO}"
      --seed "${SEED}"
    )
    if [[ "${KEEP_LAST_FRAME}" == "0" ]]; then
      args+=(--no-keep-last-frame)
    fi
  fi
  if [[ ! -f "${src}" ]]; then
    echo "错误: 未找到预处理脚本: ${src}" >&2
    exit 1
  fi
  echo "[reward-workflow] preprocess → ${out}"
  echo "[reward-workflow] $ python ${args[*]}"
  python "${args[@]}"
}

run_train() {
  local cfg="${1:-reward_training}"
  local cfg_file="${REWARD_DIR}/config/${cfg}.yaml"
  if [[ ! -f "${cfg_file}" ]]; then
    echo "错误: 训练配置不存在: ${cfg_file}" >&2
    exit 1
  fi
  local log_dir="${ROOT}/logs/$(date +'%Y%m%d-%H:%M:%S')-${cfg}"
  mkdir -p "${log_dir}"
  echo "[reward-workflow] log_dir=${log_dir}"
  local cmd=(
    python "${REWARD_DIR}/train_reward_model.py"
    --config-path "${REWARD_DIR}/config"
    --config-name "${cfg}"
    "runner.logger.log_path=${log_dir}"
  )
  if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_OVERRIDES[@]}")
  fi
  echo "[reward-workflow] train overrides: ${EXTRA_OVERRIDES[*]:-}"
  printf '%q ' "${cmd[@]}" > "${log_dir}/run_reward_training.log"
  echo >> "${log_dir}/run_reward_training.log"
  "${cmd[@]}" 2>&1 | tee -a "${log_dir}/run_reward_training.log"
  return "${PIPESTATUS[0]}"
}

run_collect() {
  local cfg="${1:-realworld_collect_dataset}"
  local cfg_file="${REWARD_DIR}/config/${cfg}.yaml"
  if [[ ! -f "${cfg_file}" ]]; then
    echo "错误: 采集配置不存在: ${cfg_file}" >&2
    exit 1
  fi
  local log_dir="${ROOT}/logs/$(date +'%Y%m%d-%H:%M:%S')-${cfg}"
  mkdir -p "${log_dir}"
  echo "[reward-workflow] log_dir=${log_dir}"
  local cmd=(
    python "${REWARD_DIR}/realworld_collect_process_dataset.py"
    --config-path "${REWARD_DIR}/config"
    --config-name "${cfg}"
    "runner.logger.log_path=${log_dir}"
  )
  if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_OVERRIDES[@]}")
  fi
  printf '%q ' "${cmd[@]}" > "${log_dir}/run_collect_process.log"
  echo >> "${log_dir}/run_collect_process.log"
  "${cmd[@]}" 2>&1 | tee -a "${log_dir}/run_collect_process.log"
  return "${PIPESTATUS[0]}"
}

_auto_data_overrides() {
  local out="$1"
  if [[ "${DATASET_TYPE}" == "qwentrend" ]]; then
    # QwenTrend 导出 jsonl + pkl；训练侧通常用不同 config，这里仍给出常见路径提示
    EXTRA_OVERRIDES+=(
      "data.train_data_paths=${out}/train.jsonl"
      "data.val_data_paths=${out}/val.jsonl"
    )
  else
    EXTRA_OVERRIDES+=(
      "data.train_data_paths=${out}/train.pt"
      "data.val_data_paths=${out}/val.pt"
    )
  fi
}

case "${MODE_NORM}" in
  preprocess|prep)
    run_preprocess "${RAW_DATA_PATH}" "${OUTPUT_DIR}"
    echo "[reward-workflow] preprocess done: ${OUTPUT_DIR}"
    ;;
  train|training)
    if [[ -z "${CONFIG_NAME}" || "${CONFIG_NAME}" == "realworld_collect_dataset" ]]; then
      CONFIG_NAME="reward_training"
    fi
    run_train "${CONFIG_NAME}"
    ;;
  collect|collection)
    if [[ -z "${CONFIG_NAME}" || "${CONFIG_NAME}" == "reward_training" ]]; then
      CONFIG_NAME="realworld_collect_dataset"
    fi
    run_collect "${CONFIG_NAME}"
    ;;
  pipeline|full)
    run_preprocess "${RAW_DATA_PATH}" "${OUTPUT_DIR}"
    _auto_data_overrides "${OUTPUT_DIR}"
    if [[ -z "${CONFIG_NAME}" || "${CONFIG_NAME}" == "realworld_collect_dataset" ]]; then
      CONFIG_NAME="reward_training"
    fi
    run_train "${CONFIG_NAME}"
    ;;
  *)
    echo "错误: --mode 应为 preprocess|train|collect|pipeline，收到: ${MODE}" >&2
    exit 1
    ;;
esac
