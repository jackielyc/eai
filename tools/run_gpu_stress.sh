#!/usr/bin/env bash
# GPU stress test wrapper — defaults to all visible GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="/share_data/projects/mahjong/share/personal/liyichao"
PYTHON="${PYTHON:-${WORKSPACE}/miniconda3/envs/Qwen2.5-VL/bin/python}"
SCRIPT="${ROOT}/tools/gpu_stress.py"

usage() {
  cat <<'EOF'
Usage: run_gpu_stress.sh [options]

Options (passed to gpu_stress.py):
  -n, --num-gpus N       Use first N GPUs (default: all)
  --gpu-ids IDS          Comma-separated GPU indices, e.g. 0,2,4
  -d, --duration SEC     Run for SEC seconds (default: until Ctrl+C)
  -s, --matrix-size N    GEMM size (0 = auto from GPU memory)
  --dtype fp16|bf16|fp32 Compute dtype (default: fp16)
  --streams N            CUDA streams per GPU (default: 4)
  -h, --help             Show this help

Environment:
  CUDA_VISIBLE_DEVICES   Limit which GPUs are visible (default: all)
  PYTHON                 Python interpreter with PyTorch+CUDA

Examples:
  bash tools/run_gpu_stress.sh                    # all GPUs, run until Ctrl+C
  bash tools/run_gpu_stress.sh -n 4              # first 4 GPUs
  bash tools/run_gpu_stress.sh --gpu-ids 0,1 -d 300
  CUDA_VISIBLE_DEVICES=2,3 bash tools/run_gpu_stress.sh -n 2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "[error] python not found: ${PYTHON}" >&2
  echo "Set PYTHON=... to a PyTorch+CUDA interpreter." >&2
  exit 1
fi

echo "[info] python=${PYTHON}"
exec "${PYTHON}" "${SCRIPT}" "$@"
