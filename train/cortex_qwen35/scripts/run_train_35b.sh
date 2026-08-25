#!/usr/bin/env bash
# LoRA SFT Qwen3.5-35B-A3B on Cortex System-2
# Default: single-process device_map=auto across visible GPUs (fits MoE better than naive DDP).
# Optional FSDP: DEVICE_MAP=none FSDP=full_shard NPROC=8 bash run_train_35b.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/qwen35_35b_a3b_lora.yaml}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
FSDP="${FSDP:-none}"
NPROC="${NPROC:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK=1

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/train_35b_$(date +%Y%m%d_%H%M%S).log"

echo "[info] python=${PYTHON}"
echo "[info] config=${CONFIG}"
echo "[info] device_map=${DEVICE_MAP} fsdp=${FSDP}"
echo "[info] gpus=${CUDA_VISIBLE_DEVICES}"
echo "[info] log=${LOG}"

if [[ "${DEVICE_MAP}" == "auto" ]]; then
  "${PYTHON}" "${ROOT}/scripts/train_lora_sft.py" \
    --config "${CONFIG}" \
    --device_map auto \
    2>&1 | tee "${LOG}"
else
  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    "${ROOT}/scripts/train_lora_sft.py" \
    --config "${CONFIG}" \
    --device_map none \
    --fsdp "${FSDP}" \
    2>&1 | tee "${LOG}"
fi
