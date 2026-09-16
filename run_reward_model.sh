#!/usr/bin/env bash
# 在独立 Python 进程中跑 RLinf reward 打分（避免污染 viewer 的 ROS PYTHONPATH）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/reward_model_worker.py"
ROOT="${RLINF_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/RLinf}"
DEFAULT_PY="/home/psibot/miniconda3/envs/RLinf/bin/python"
FALLBACK_PY="/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/RLinf/bin/python"
PY="${RLINF_PYTHON:-}"

unset PYTHONPATH PYTHONHOME || true
export PYTHONNOUSERSITE=1
export RLINF_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PY" ]]; then
  if [[ -x "$DEFAULT_PY" ]]; then
    PY="$DEFAULT_PY"
  elif [[ -x "$FALLBACK_PY" ]]; then
    PY="$FALLBACK_PY"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "错误: 找不到 Python。请设置 RLINF_PYTHON" >&2
    exit 1
  fi
fi

if [[ ! -f "$WORKER" ]]; then
  echo "错误: 未找到 $WORKER" >&2
  exit 1
fi
if [[ ! -d "$ROOT/rlinf/models/embodiment/reward" ]]; then
  echo "错误: RLinf reward 包不存在: $ROOT" >&2
  exit 1
fi

exec "$PY" -u "$WORKER" --rlinf-root "$ROOT" "$@"
