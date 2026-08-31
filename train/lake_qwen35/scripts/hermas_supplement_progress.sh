#!/usr/bin/env bash
# Print one-line Hermes supplement export progress summary.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
OUT="${ROOT}/output"
LOG="$(ls -t "${OUT}"/convert_hermas_supplement_*.log 2>/dev/null | head -1 || true)"
PID_FILE="${OUT}/convert_hermas_supplement.pid"
APPROVED_TRAIN=3769067
APPROVED_VAL=418845
MAIN_TRAIN=8926
MAIN_VAL=8761

worker_pid=""
if [[ -f "${PID_FILE}" ]]; then
  worker_pid="$(cat "${PID_FILE}")"
fi

if [[ -n "${worker_pid}" ]] && ps -p "${worker_pid}" >/dev/null 2>&1; then
  proc_state="running pid=${worker_pid} $(ps -p "${worker_pid}" -o etime= | tr -d ' ')"
else
  proc_state="stopped"
fi

sup_train=0
sup_val=0
[[ -f "${DATA}/hermas_sys2_train_supplement.jsonl" ]] && sup_train=$(wc -l < "${DATA}/hermas_sys2_train_supplement.jsonl")
[[ -f "${DATA}/hermas_sys2_val_supplement.jsonl" ]] && sup_val=$(wc -l < "${DATA}/hermas_sys2_val_supplement.jsonl")

remain_train=$((APPROVED_TRAIN - MAIN_TRAIN - sup_train))
remain_val=$((APPROVED_VAL - MAIN_VAL - sup_val))
if (( remain_train < 0 )); then remain_train=0; fi
if (( remain_val < 0 )); then remain_val=0; fi
pct_train=$(python3 - <<PY
sup=${sup_train}; main=${MAIN_TRAIN}; total=${APPROVED_TRAIN}
print(f"{(sup+main)/total*100:.2f}")
PY
)

last_line=""
rate=""
if [[ -n "${LOG}" && -f "${LOG}" ]]; then
  last_line="$(rg '\[progress\]|\[scan\]|\[ok\]' "${LOG}" | tail -1 || true)"
  rate="$(sed -n 's/.*(\([0-9.]*\) clips\/s).*/\1/p' <<<"${last_line}" | tail -1)"
fi

eta="?"
if [[ -n "${rate}" && "${rate}" != "0" ]]; then
  eta="$(python3 - <<PY
remain=${remain_train}+${remain_val}
rate=float("${rate}")
sec=remain/rate
h=int(sec//3600); m=int((sec%3600)//60)
print(f"~{h}h{m}m" if h else f"~{m}m")
PY
)"
fi

echo "[hermas-supplement] ${proc_state} | train=${sup_train}/${APPROVED_TRAIN} (${pct_train}%) val_sup=${sup_val} | remain train=${remain_train} val=${remain_val} | rate=${rate:-?} clips/s eta=${eta}"
[[ -n "${last_line}" ]] && echo "  last: ${last_line}"
