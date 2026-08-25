#!/usr/bin/env bash
# Convert share_data_lake view -> multimodal ShareGPT JSON + exported RGB frames
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
# zarr 读取依赖 psi-policy 环境；训练仍用 Qwen2.5-VL
CONVERT_PYTHON="${CONVERT_PYTHON:-${WORKSPACE}/miniconda3/envs/psi-policy/bin/python}"

DATA_ROOT="${DATA_ROOT:-/share_data_lake}"
DATAHOUSE_ID="${DATAHOUSE_ID:-hermes-human-ego-10029}"
VIEW_ID="${VIEW_ID:-10029-hermes-data-3_VA48DX}"
CAMERA="${CAMERA:-auto}"
SUBSET_TRAIN="${SUBSET_TRAIN:-20000}"
SUBSET_VAL="${SUBSET_VAL:-2000}"
TASKS="${TASKS:-}"
TASK_LIST_FILE="${TASK_LIST_FILE:-}"
MAX_PER_TASK="${MAX_PER_TASK:-}"
NUM_WORKERS="${NUM_WORKERS:-$(nproc)}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-1}"
SKIP_FULL="${SKIP_FULL:-1}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --datahouse-id "${DATAHOUSE_ID}"
  --view-id "${VIEW_ID}"
  --output-dir "${ROOT}/data"
  --camera "${CAMERA}"
  --subset-train "${SUBSET_TRAIN}"
  --subset-val "${SUBSET_VAL}"
)
if [[ -n "${TASKS}" ]]; then
  ARGS+=(--tasks "${TASKS}")
fi
if [[ -n "${TASK_LIST_FILE}" ]]; then
  ARGS+=(--task-list-file "${TASK_LIST_FILE}")
fi
if [[ -n "${MAX_PER_TASK}" ]]; then
  ARGS+=(--max-per-task "${MAX_PER_TASK}")
fi
ARGS+=(--num-workers "${NUM_WORKERS}")
if [[ "${SKIP_EXISTING}" == "0" ]]; then
  ARGS+=(--no-skip-existing)
fi
if [[ "${RESUME}" == "0" ]]; then
  ARGS+=(--no-resume)
fi
if [[ -n "${SKIP_FULL}" && "${SKIP_FULL}" != "0" ]]; then
  ARGS+=(--skip-full)
fi

echo "[info] convert ${DATAHOUSE_ID}/${VIEW_ID} -> ${ROOT}/data"
"${CONVERT_PYTHON}" "${ROOT}/scripts/convert_lake_to_sft.py" "${ARGS[@]}"
