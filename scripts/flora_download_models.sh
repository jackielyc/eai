#!/usr/bin/env bash
# Download FLORA suite into .../liyichao/models/
# Prefer: scripts/flora_download_models.py (resume-friendly curl for large shards)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${FLORA_MODELS_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/models}"
PY="${FLORA_DOWNLOAD_PYTHON:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/aspire/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${MODELS}/Flora/hf_home"
mkdir -p "$HF_HOME" "$MODELS" "$ROOT/log"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "ROOT=$ROOT"

download() {
  local repo="$1" dest="$2"
  echo "==== $(date) $repo -> $dest ===="
  mkdir -p "$dest"
  local attempt=1
  while true; do
    echo "-- attempt ${attempt} $(date) --"
    if "$PY" - "$repo" "$dest" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], Path(sys.argv[2])
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
path = snapshot_download(
    repo_id=repo,
    local_dir=str(dest),
    token=token,
    max_workers=4,
)
print("OK", path, flush=True)
PY
    then
      break
    fi
    echo "download failed for $repo (attempt ${attempt}), retry in 30s…"
    attempt=$((attempt + 1))
    sleep 30
  done
}

download nvidia/Flora-TempSeg "$MODELS/Flora-TempSeg"
download nvidia/Flora-Captioner "$MODELS/Flora-Captioner"
download nvidia/Flora-Rewarder "$MODELS/Flora-Rewarder"

if [[ ! -f "$MODELS/Qwen/Qwen3.5-9B/config.json" ]]; then
  download Qwen/Qwen3.5-9B "$MODELS/Qwen/Qwen3.5-9B"
else
  echo "skip Qwen3.5-9B (already present)"
fi

echo "==== ALL DONE $(date) ===="
du -sh "$MODELS/Flora-TempSeg" "$MODELS/Flora-Captioner" "$MODELS/Flora-Rewarder" \
  "$MODELS/Qwen/Qwen3.5-9B" 2>/dev/null || true
