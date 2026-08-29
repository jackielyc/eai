#!/usr/bin/env bash
# Export remaining Hermes clips into *_supplement.jsonl, keeping existing hermas_sys2_*.jsonl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=convert_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/convert_env.sh"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
CONVERT_PYTHON="${CONVERT_PYTHON:-${WORKSPACE}/miniconda3/envs/psi-policy/bin/python}"

DATA_ROOT="${DATA_ROOT:-/share_data_lake}"
DATAHOUSE_ID="${DATAHOUSE_ID:-hermes-human-ego-10029}"
VIEW_ID="${VIEW_ID:-10029-hermes-data-3_VA48DX}"
CAMERA="${CAMERA:-auto}"
NUM_WORKERS="$(default_num_workers)"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RESUME="${RESUME:-1}"

ARGS=(
  --data-root "${DATA_ROOT}"
  --datahouse-id "${DATAHOUSE_ID}"
  --view-id "${VIEW_ID}"
  --output-dir "${ROOT}/data"
  --camera "${CAMERA}"
  --num-workers "${NUM_WORKERS}"
  --supplement
)
if [[ "${SKIP_EXISTING}" == "0" ]]; then
  ARGS+=(--no-skip-existing)
fi
if [[ "${RESUME}" == "0" ]]; then
  ARGS+=(--no-resume)
fi

mkdir -p "${ROOT}/output"
LOG="${ROOT}/output/convert_hermas_supplement_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${ROOT}/output/convert_hermas_supplement.pid"

echo "[info] keep existing: ${ROOT}/data/hermas_sys2_{train,val}.jsonl"
echo "[info] export remainder -> ${ROOT}/data/hermas_sys2_{train,val}_supplement.jsonl"
echo "[info] workers=${NUM_WORKERS} resume=${RESUME} skip_existing=${SKIP_EXISTING}"
echo "[info] log=${LOG}"

nohup "${CONVERT_PYTHON}" "${ROOT}/scripts/convert_lake_to_sft.py" "${ARGS[@]}" >"${LOG}" 2>&1 &
echo $! >"${PID_FILE}"
echo "[info] started pid=$(cat "${PID_FILE}")"
