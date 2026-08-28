#!/usr/bin/env bash
# Multimodal LoRA SFT Qwen3.5-4B on share_data_lake (single- or multi-node DDP)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/qwen35_4b_lora.yaml}"

# shellcheck source=dist_env.sh
source "${ROOT}/scripts/dist_env.sh"
dist_spawn_workers "${BASH_SOURCE[0]}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/train_4b_n${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

dist_ensure_dataset

TRAIN_EXTRA_ARGS=()
if [[ "${FRESH:-}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--no-auto_resume)
elif [[ "${RESUME:-}" == "1" ]]; then
  TRAIN_EXTRA_ARGS+=(--resume_from_checkpoint auto)
fi

echo "[info] python=${PYTHON}"
echo "[info] config=${CONFIG}"
echo "[info] nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[info] nproc=${NPROC} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[info] log=${LOG}"

"${DIST_LAUNCH[@]}" \
  "${ROOT}/scripts/train_lora_sft_mm.py" \
  --config "${CONFIG}" \
  "${TRAIN_EXTRA_ARGS[@]+"${TRAIN_EXTRA_ARGS[@]}"}" \
  2>&1 | tee -i "${LOG}"
