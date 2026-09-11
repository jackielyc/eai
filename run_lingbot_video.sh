#!/usr/bin/env bash
# 在独立 Python 进程中跑 LingBot-Video（避免污染 viewer 的 ROS PYTHONPATH）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/lingbot_video_worker.py"
ROOT="${LINGBOT_VIDEO_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/lingbot-video}"
DEFAULT_PY="/home/psibot/miniconda3/envs/eai/bin/python"
PY="${LINGBOT_VIDEO_PYTHON:-}"

unset PYTHONPATH PYTHONHOME || true
export PYTHONNOUSERSITE=1
export LINGBOT_VIDEO_ROOT="$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rewriter${PYTHONPATH:+:$PYTHONPATH}"
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_flash}"
# Qwen3-VL text encoder：无 flash-attn3 时用 sdpa（runner 读此环境变量）
export LINGBOT_QWEN_ATTN_IMPLEMENTATION="${LINGBOT_QWEN_ATTN_IMPLEMENTATION:-sdpa}"

if [[ -d /usr/local/cuda/compat ]]; then
  export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "$PY" ]]; then
  if [[ -x "$DEFAULT_PY" ]]; then
    PY="$DEFAULT_PY"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 LINGBOT_VIDEO_PYTHON" >&2
    exit 1
  fi
fi

if [[ ! -f "$WORKER" ]]; then
  echo "错误: 未找到 $WORKER" >&2
  exit 1
fi
if [[ ! -f "$ROOT/scripts/inference.py" ]]; then
  echo "错误: lingbot-video 仓库不存在: $ROOT" >&2
  exit 1
fi

exec "$PY" -u "$WORKER" "$@"
