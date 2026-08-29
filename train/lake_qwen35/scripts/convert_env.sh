#!/usr/bin/env bash
# Shared helpers for lake/Hermes/VN data conversion scripts.

default_num_workers() {
  if [[ -n "${NUM_WORKERS:-}" ]]; then
    echo "${NUM_WORKERS}"
    return 0
  fi
  if [[ -n "${WORKERS:-}" ]]; then
    echo "${WORKERS}"
    return 0
  fi

  local python_bin="${CONVERT_PYTHON:-${PYTHON:-python3}}"
  local n
  n="$("${python_bin}" - <<'PY' 2>/dev/null || true
import os
try:
    n = len(os.sched_getaffinity(0))
except Exception:
    n = os.cpu_count() or 1
print(max(1, int(n)))
PY
)"
  if [[ -n "${n}" && "${n}" =~ ^[0-9]+$ && "${n}" -ge 1 ]]; then
    echo "${n}"
    return 0
  fi

  n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  if [[ -z "${n}" || "${n}" -lt 1 ]]; then
    n="$(nproc --all 2>/dev/null || nproc 2>/dev/null || true)"
  fi
  if [[ -z "${n}" || "${n}" -lt 1 ]]; then
    n="$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 1)"
  fi
  echo "${n:-1}"
}
