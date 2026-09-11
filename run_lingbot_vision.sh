#!/usr/bin/env bash
# 在独立 Python 进程中跑 LingBot-Vision PCA（避免污染 viewer 的 ROS PYTHONPATH）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/lingbot_vision_worker.py"
ROOT="${LINGBOT_VISION_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/lingbot-vision}"
DEFAULT_PY="/home/psibot/miniconda3/envs/eai/bin/python"
PY="${LINGBOT_VISION_PYTHON:-}"

unset PYTHONPATH PYTHONHOME || true
export PYTHONNOUSERSITE=1
export LINGBOT_VISION_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PY" ]]; then
  if [[ -x "$DEFAULT_PY" ]]; then
    PY="$DEFAULT_PY"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 LINGBOT_VISION_PYTHON" >&2
    exit 1
  fi
fi

if [[ ! -f "$WORKER" ]]; then
  echo "错误: 未找到 $WORKER" >&2
  exit 1
fi
if [[ ! -f "$ROOT/lingbot_vision/__init__.py" ]]; then
  echo "错误: lingbot-vision 仓库不存在: $ROOT" >&2
  exit 1
fi

exec "$PY" -u "$WORKER" "$@"
