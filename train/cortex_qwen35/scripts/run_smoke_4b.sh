#!/usr/bin/env bash
# Tiny smoke test on 1 GPU (few samples, 1 epoch)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
GPU="${GPU:-0}"
OUT="${ROOT}/output/qwen35-4b-lora-smoke"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/smoke_$(date +%Y%m%d_%H%M%S).log"

"${PYTHON}" "${ROOT}/scripts/train_lora_sft.py" \
  --model_name_or_path "${WORKSPACE}/models/Qwen/Qwen3.5-4B" \
  --dataset_path "${ROOT}/data/cortex_sys2_train_20k.json" \
  --eval_dataset_path "${ROOT}/data/cortex_sys2_val_2k.json" \
  --output_dir "${OUT}" \
  --max_samples 32 \
  --eval_max_samples 8 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 1 \
  --logging_steps 1 \
  --save_steps 50 \
  --eval_steps 50 \
  --lora_rank 8 \
  2>&1 | tee "${LOG}"
