#!/usr/bin/env bash
# Multimodal LoRA SFT Qwen3.5-4B on share_data_lake (multi-GPU DDP)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/qwen35_4b_lora.yaml}"
NPROC="${NPROC:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/train_4b_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${ROOT}/data/lake_sys2_train_20k.json" && ! -f "${ROOT}/data/lake_sys2_train.jsonl" ]]; then
  echo "[info] dataset missing, running convert first..."
  bash "${ROOT}/scripts/run_convert.sh"
fi

echo "[info] python=${PYTHON}"
echo "[info] config=${CONFIG}"
echo "[info] nproc=${NPROC} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[info] log=${LOG}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  "${ROOT}/scripts/train_lora_sft_mm.py" \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG}"
