#!/usr/bin/env bash
# Tiny smoke test: multimodal lake SFT on 1 GPU
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
GPU="${GPU:-0}"
OUT="${ROOT}/output/qwen35-4b-lora-lake-smoke"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/smoke_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${ROOT}/data/hermas_sys2_train_20k.json" ]]; then
  echo "[info] dataset missing, running convert first..."
  SUBSET_TRAIN=64 SUBSET_VAL=16 SKIP_FULL=1 bash "${ROOT}/scripts/run_convert.sh"
fi

"${PYTHON}" -u "${ROOT}/scripts/train_lora_sft_mm.py" \
  --model_name_or_path "${WORKSPACE}/models/Qwen/Qwen3.5-4B" \
  --dataset_path "${ROOT}/data/hermas_sys2_train_20k.json" \
  --eval_dataset_path "${ROOT}/data/hermas_sys2_val_2k.json" \
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
