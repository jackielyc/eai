#!/usr/bin/env bash
# 在独立 Python 进程中跑 LingBot-World-V2（优先用仓库自带 .venv）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/lingbot_world_worker.py"
ROOT="${LINGBOT_WORLD_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/lingbot-world-v2}"
VENV_PY="${ROOT}/.venv/bin/python"
EAI_PY="/home/psibot/miniconda3/envs/eai/bin/python"
PY="${LINGBOT_WORLD_PYTHON:-}"

unset PYTHONPATH PYTHONHOME || true
export PYTHONNOUSERSITE=1
export LINGBOT_WORLD_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PY" ]]; then
  if [[ -x "$VENV_PY" ]]; then
    PY="$VENV_PY"
  elif [[ -x "$EAI_PY" ]]; then
    PY="$EAI_PY"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 LINGBOT_WORLD_PYTHON" >&2
    exit 1
  fi
fi

if [[ ! -f "$WORKER" ]]; then
  echo "错误: 未找到 $WORKER" >&2
  exit 1
fi
if [[ ! -f "$ROOT/generate.py" ]]; then
  echo "错误: lingbot-world-v2 仓库不存在: $ROOT" >&2
  exit 1
fi

echo "[lingbot_world] python=$PY" >&2
exec "$PY" -u "$WORKER" "$@"
