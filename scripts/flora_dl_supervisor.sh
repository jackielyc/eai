#!/usr/bin/env bash
# Supervise Flora model download; logs stay under eai/log/.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/log/flora_model_download.log"
DONE="$ROOT/log/flora_download.DONE"
PY="${FLORA_DOWNLOAD_PYTHON:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/aspire/bin/python}"
SCRIPT="$ROOT/scripts/flora_download_models.py"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$ROOT/log"
attempt=0
while true; do
  if grep -q '==== ALL DONE' "$LOG" 2>/dev/null; then
    echo "supervisor saw ALL DONE" | tee -a "$LOG"
    date >"$DONE"
    du -sh /share_data/projects/mahjong/share/personal/liyichao/models/Flora-* \
      /share_data/projects/mahjong/share/personal/liyichao/models/Qwen/Qwen3.5-9B >>"$DONE" 2>/dev/null || true
    exit 0
  fi
  attempt=$((attempt+1))
  printf '\n==== %s supervisor start attempt %s ====\n' "$(date)" "$attempt" >>"$LOG"
  "$PY" -u "$SCRIPT" >>"$LOG" 2>&1
  code=$?
  printf '==== %s child exit=%s ====\n' "$(date)" "$code" >>"$LOG"
  if grep -q '==== ALL DONE' "$LOG" 2>/dev/null; then
    date >"$DONE"
    du -sh /share_data/projects/mahjong/share/personal/liyichao/models/Flora-* \
      /share_data/projects/mahjong/share/personal/liyichao/models/Qwen/Qwen3.5-9B >>"$DONE" 2>/dev/null || true
    exit 0
  fi
  sleep 15
done
