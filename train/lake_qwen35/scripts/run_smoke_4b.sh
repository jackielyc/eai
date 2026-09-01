#!/usr/bin/env bash
# Tiny smoke test: multimodal Hermes SFT on 1 GPU
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
GPU="${GPU:-0}"
OUT="${ROOT}/output/qwen35-4b-lora-hermas-approved-smoke"
TRAIN_JSONL="${ROOT}/data/hermas_sys2_train_approved.jsonl"
VAL_JSONL="${ROOT}/data/hermas_sys2_val_approved.jsonl"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/smoke_hermas_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[error] missing ${TRAIN_JSONL}; export Hermes approved data first" >&2
  exit 1
fi

"${PYTHON}" -u "${ROOT}/scripts/train_lora_sft_mm.py" \
  --model_name_or_path "${WORKSPACE}/models/Qwen/Qwen3.5-4B" \
  --dataset_path "${TRAIN_JSONL}" \
  --eval_dataset_path "${VAL_JSONL}" \
  --output_dir "${OUT}" \
  --max_samples 16 \
  --eval_max_samples 4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 1 \
  --logging_steps 1 \
  --save_steps 20 \
  --eval_steps 20 \
  --lora_rank 8 \
  --image_max_pixels 131072 \
  --dataloader_num_workers 0 \
  2>&1 | tee -i "${LOG}"
