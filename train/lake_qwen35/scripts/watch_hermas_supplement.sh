#!/usr/bin/env bash
# Live progress for Hermes supplement export.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/data"
LOG="$(ls -t "${ROOT}/output"/convert_hermas_supplement_*.log 2>/dev/null | head -1 || true)"
PID_FILE="${ROOT}/output/convert_hermas_supplement.pid"
TRAIN_TOTAL=5350879
VAL_TOTAL=594098
EXCLUDE_TRAIN=14803
EXCLUDE_VAL=14036

train_lines() {
  wc -l <"${DATA}/hermas_sys2_train_supplement.jsonl" 2>/dev/null | tr -d ' ' || echo 0
}

val_lines() {
  if [[ -f "${DATA}/hermas_sys2_val_supplement.jsonl" ]]; then
    wc -l <"${DATA}/hermas_sys2_val_supplement.jsonl" | tr -d ' '
  else
    echo 0
  fi
}

show_once() {
  local train val pid status rate remain eta
  train="$(train_lines)"
  val="$(val_lines)"
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    status="running pid=${pid} elapsed=$(ps -p "${pid}" -o etime= | tr -d ' ')"
  else
    status="NOT RUNNING"
  fi
  rate=""
  if [[ -n "${LOG}" && -f "${LOG}" ]]; then
    rate="$(grep -oE '[0-9]+\.[0-9]+ clips/s' "${LOG}" | tail -1 | awk '{print $1}' || true)"
    echo "[log] ${LOG}"
    tail -3 "${LOG}" | sed 's/^/  /'
  fi
  remain=$((TRAIN_TOTAL - EXCLUDE_TRAIN - train))
  [[ "${remain}" -lt 0 ]] && remain=0
  printf '\n[train supplement] %s / %s (remain %s)\n' "${train}" "$((TRAIN_TOTAL - EXCLUDE_TRAIN))" "${remain}"
  printf '[val supplement]   %s\n' "${val}"
  printf '[status] %s\n' "${status}"
  if [[ -n "${rate}" && "${remain}" -gt 0 ]]; then
    eta="$(python3 - <<PY
rate=float("${rate}")
remain=${remain}
print(f'{remain/rate/3600:.1f}h')
PY
)"
    printf '[speed]  %s clips/s  ETA(train) ~%s\n' "${rate}" "${eta}"
  fi
  echo
}

if [[ "${1:-}" == "--once" ]]; then
  show_once
  exit 0
fi

echo "Watching Hermes supplement export (Ctrl-C to stop watching; job keeps running)"
while true; do
  clear || true
  date '+%F %T'
  show_once
  sleep "${INTERVAL:-10}"
done
