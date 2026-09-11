#!/usr/bin/env bash
# 在独立 Python 进程中跑 LingBot-Map demo（viser 3D 重建可视化）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/lingbot_map_worker.py"
ROOT="${LINGBOT_MAP_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/lingbot-map}"
DEFAULT_PY="/home/psibot/miniconda3/envs/eai/bin/python"
PY="${LINGBOT_MAP_PYTHON:-}"

unset PYTHONPATH PYTHONHOME || true
export PYTHONNOUSERSITE=1
export LINGBOT_MAP_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PY" ]]; then
  if [[ -x "$DEFAULT_PY" ]]; then
    PY="$DEFAULT_PY"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 LINGBOT_MAP_PYTHON" >&2
    exit 1
  fi
fi

if [[ ! -f "$WORKER" ]]; then
  echo "错误: 未找到 $WORKER" >&2
  exit 1
fi
if [[ ! -f "$ROOT/demo.py" ]]; then
  echo "错误: lingbot-map 仓库不存在: $ROOT" >&2
  exit 1
fi

exec "$PY" -u "$WORKER" "$@"
