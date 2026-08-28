#!/usr/bin/env bash
# 本地部署 Qwen3.5（4B / 35B-A3B 等）为 OpenAI 兼容推理服务
#
# 默认权重: ../models/Qwen/Qwen3.5-4B
# 默认 API: http://127.0.0.1:8100/v1  model=qwen3.5-4b
#
# 与 show_camera_topics.py「测试」Tab:
#   Docker viewer 会经宿主机 local_qwen_hostctl 启动本脚本。
#   也可手动:
#     bash run_local_qwen.sh
#     bash run_local_qwen.sh --host 0.0.0.0
#     bash run_local_qwen.sh --model ../models/Qwen/Qwen3.5-35B-A3B --model-id qwen3.5-35b-a3b
#     bash run_local_qwen.sh --check
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/local_qwen_worker.py"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${LOCAL_QWEN_MODEL_DIR:-${WORKSPACE_DIR}/models/Qwen/Qwen3.5-4B}"
PORT="${LOCAL_QWEN_PORT:-8100}"
HOST="${LOCAL_QWEN_HOST:-127.0.0.1}"
MODEL_ID="${LOCAL_QWEN_MODEL_ID:-qwen3.5-4b}"
PY="${LOCAL_QWEN_PYTHON:-}"

DO_CHECK=0
LAZY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1; shift ;;
    --lazy) LAZY=1; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL_DIR="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# Docker 内 HOME 常为 /root，必须用绝对路径找宿主机 conda
resolve_python() {
  local cand
  for cand in \
    "${PY}" \
    "${LOCAL_QWEN_PYTHON:-}" \
    "/home/psibot/miniconda3/envs/psi-policy/bin/python" \
    "${PSIBOT_HOME:+${PSIBOT_HOME}/miniconda3/envs/psi-policy/bin/python}" \
    "${HOME}/miniconda3/envs/psi-policy/bin/python" \
    "/home/psibot/miniconda3/envs/robotics/bin/python" \
    "${HOME}/miniconda3/envs/robotics/bin/python" \
    "$(command -v python3 || true)"
  do
    [[ -z "${cand}" ]] && continue
    [[ -x "${cand}" ]] || continue
    # 只要能 import；CUDA 在无驱动容器内可能为 False，但仍可用于宿主机启动路径校验
    if "${cand}" -c "import torch, transformers" >/dev/null 2>&1; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${PY}" ]]; then
  PY="$(resolve_python || true)"
fi

if [[ "$DO_CHECK" == "1" ]]; then
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "OK: Local Qwen 在线 http://127.0.0.1:${PORT}/v1  model=${MODEL_ID}"
    exit 0
  fi
  echo "离线: http://127.0.0.1:${PORT}/v1"
  exit 1
fi

if [[ -z "${PY}" || ! -x "${PY}" ]]; then
  echo "错误: 未找到带 torch+transformers 的 Python。" >&2
  echo "请设置: LOCAL_QWEN_PYTHON=/home/psibot/miniconda3/envs/psi-policy/bin/python" >&2
  exit 1
fi

if [[ ! -f "${WORKER}" ]]; then
  echo "错误: 未找到 ${WORKER}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" && ! -f "${MODEL_DIR}/adapter_config.json" ]]; then
  echo "错误: 未找到模型目录 ${MODEL_DIR}（需 config.json 或 adapter_config.json）" >&2
  exit 1
fi

MODEL_NAME="$(basename "${MODEL_DIR}")"
echo ">>> 启动本地 ${MODEL_NAME} 推理服务"
echo "    PYTHON=${PY}"
echo "    MODEL_DIR=${MODEL_DIR}"
echo "    API: http://${HOST}:${PORT}/v1   model=${MODEL_ID}"
echo "    对话面板可选: 本地 Qwen 服务"
"${PY}" -c "import torch; print('    CUDA:', torch.cuda.is_available(), 'devices=', torch.cuda.device_count())" 2>/dev/null || true

export LOCAL_QWEN_MODEL_DIR="${MODEL_DIR}"
export LOCAL_QWEN_MODEL_ID="${MODEL_ID}"
export LOCAL_QWEN_HOST="${HOST}"
export LOCAL_QWEN_PORT="${PORT}"

ARGS=(--host "${HOST}" --port "${PORT}" --model "${MODEL_DIR}" --model-id "${MODEL_ID}")
if [[ "${LAZY}" == "1" ]]; then
  ARGS+=(--lazy)
fi

exec "${PY}" "${WORKER}" "${ARGS[@]}"
