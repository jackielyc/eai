#!/usr/bin/env bash
# Steinate/Cortex VN norm_mem -> multimodal vn_sys2 JSONL + ego frames from COS MCAP
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=convert_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/convert_env.sh"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"

CORTEX_DIR="${CORTEX_DIR:-${WORKSPACE}/dataset/Steinate/Cortex}"
WORKERS="$(default_num_workers)"
SPLITS="${SPLITS:-train,val}"
LIMIT="${LIMIT:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-1}"
PREFER_LOCAL="${PREFER_LOCAL:-1}"

ARGS=(
  --cortex-dir "${CORTEX_DIR}"
  --output-dir "${ROOT}/data"
  --workers "${WORKERS}"
  --splits "${SPLITS}"
)
if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ "${SKIP_EXISTING}" == "0" ]]; then
  ARGS+=(--no-skip-existing)
fi
if [[ "${RESUME}" == "0" ]]; then
  ARGS+=(--no-resume)
fi
if [[ "${PREFER_LOCAL}" == "0" ]]; then
  ARGS+=(--no-prefer-local)
fi

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/attach_vn_images_$(date +%Y%m%d_%H%M%S).log"

echo "[info] cortex=${CORTEX_DIR}"
echo "[info] output=${ROOT}/data/vn_sys2_{train,val}.jsonl"
echo "[info] workers=${WORKERS} splits=${SPLITS} resume=${RESUME} prefer_local=${PREFER_LOCAL}"
echo "[info] log=${LOG}"

"${PYTHON}" "${ROOT}/scripts/attach_vn_images.py" "${ARGS[@]}" 2>&1 | tee -i "${LOG}"
