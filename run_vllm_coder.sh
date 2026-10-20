#!/usr/bin/env bash
# Serve Qwen3-Coder-30B-A3B with vLLM native tool-calling (option A).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${VLLM_PYTHON:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/vllm/bin/python}"
MODEL="${MODEL_DIR:-/share_data/projects/mahjong/share/personal/liyichao/models/Qwen/Qwen3-Coder-30B-A3B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen3-coder-30b-a3b}"
PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
# EmbodiedEvo/Codex 系统提示 + instruction.md 很容易顶满 16k；默认抬到 32k。
MAX_LEN="${MAX_MODEL_LEN:-32768}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
LOG="${LOG:-$ROOT/log/persist/vllm_coder.log}"
PIDF="${PIDF:-$ROOT/log/persist/vllm_coder.pid}"

mkdir -p "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ "${1:-}" == "stop" ]]; then
  if [[ -f "$PIDF" ]]; then
    kill "$(cat "$PIDF")" 2>/dev/null || true
    rm -f "$PIDF"
  fi
  pkill -f "vllm.entrypoints.openai.api_server.*${SERVED_NAME}" 2>/dev/null || true
  echo "stopped"
  exit 0
fi

if ss -ltn | grep -q ":${PORT} "; then
  echo "port ${PORT} already in use" >&2
  exit 1
fi

setsid env PYTHONUNBUFFERED=1 \
  "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --enable-auto-tool-choice \
  --tool-call-parser "$PARSER" \
  >>"$LOG" 2>&1 </dev/null &
echo $! >"$PIDF"
echo "started pid=$(cat "$PIDF") http://${HOST}:${PORT}/v1 log=$LOG"
