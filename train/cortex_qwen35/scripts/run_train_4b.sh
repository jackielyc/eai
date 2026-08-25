#!/usr/bin/env bash
# LoRA SFT Qwen3.5-4B on Cortex System-2 (multi-GPU DDP)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/qwen35_4b_lora.yaml}"
NPROC="${NPROC:-6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/train_4b_$(date +%Y%m%d_%H%M%S).log"

echo "[info] python=${PYTHON}"
echo "[info] config=${CONFIG}"
echo "[info] nproc=${NPROC} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[info] log=${LOG}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${ROOT}/scripts/train_lora_sft.py" \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG}"
