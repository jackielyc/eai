#!/usr/bin/env bash
# Multimodal LoRA SFT Qwen3.5-35B-A3B on share_data_lake
# Single machine (default): one process, device_map=auto.
# Multi-node: torchrun + DEVICE_MAP=none (device_map=auto is single-process only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/qwen35_35b_a3b_lora.yaml}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

# shellcheck source=dist_env.sh
source "${ROOT}/scripts/dist_env.sh"
if [[ "${NNODES}" -gt 1 && "${DEVICE_MAP}" == "auto" ]]; then
  echo "[error] 35B multi-node does not support device_map=auto (single-process only)." >&2
  echo "[error] Set DEVICE_MAP=none (and FSDP in yaml/cli if needed)." >&2
  exit 1
fi
dist_spawn_workers "${BASH_SOURCE[0]}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/train_35b_n${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

dist_ensure_dataset

echo "[info] python=${PYTHON}"
echo "[info] config=${CONFIG}"
echo "[info] device_map=${DEVICE_MAP}"
echo "[info] nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[info] nproc=${NPROC} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[info] log=${LOG}"

if [[ "${NNODES}" -gt 1 ]]; then
  if [[ "${DEVICE_MAP}" == "auto" ]]; then
    echo "[error] 35B multi-node does not support device_map=auto (single-process only)." >&2
    echo "[error] Set DEVICE_MAP=none (and FSDP in yaml/cli if needed)." >&2
    exit 1
  fi
  "${DIST_LAUNCH[@]}" \
    "${ROOT}/scripts/train_lora_sft_mm.py" \
    --config "${CONFIG}" \
    --device_map "${DEVICE_MAP}" \
    2>&1 | tee -i "${LOG}"
else
  "${PYTHON}" -u "${ROOT}/scripts/train_lora_sft_mm.py" \
    --config "${CONFIG}" \
    --device_map "${DEVICE_MAP}" \
    2>&1 | tee -i "${LOG}"
fi
