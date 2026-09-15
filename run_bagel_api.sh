#!/usr/bin/env bash
# 单独部署 Bagel 模型推理 API（无 Gradio）
#
# 用法:
#   bash run_bagel_api.sh
#   bash run_bagel_api.sh --mode 2 --port 7861
#   bash run_bagel_api.sh --check
#
# 环境变量:
#   BAGEL_DIR          Bagel 仓库根目录
#   BAGEL_PYTHON       Python（推荐 eai/.cache/bagel_venv）
#   BAGEL_MODEL_PATH   权重目录
#   BAGEL_API_HOST     默认 127.0.0.1
#   BAGEL_API_PORT     默认 7861（避开 Gradio 7860）
#   BAGEL_MODE         1=bf16 / 2=NF4 / 3=INT8
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_DIR="${BAGEL_DIR:-/share_data/projects/mahjong/share/personal/liyichao/Bagel}"
BAGEL_PYTHON="${BAGEL_PYTHON:-${SCRIPT_DIR}/.cache/bagel_venv/bin/python}"
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-${SCRIPT_DIR}/.cache/bagel_models/BAGEL-7B-MoT}"
BAGEL_API_HOST="${BAGEL_API_HOST:-127.0.0.1}"
BAGEL_API_PORT="${BAGEL_API_PORT:-7861}"
BAGEL_MODE="${BAGEL_MODE:-1}"

DO_CHECK=0
EXTRA_ARGS=()

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --check) DO_CHECK=1; shift ;;
    --host) BAGEL_API_HOST="$2"; shift 2 ;;
    --port) BAGEL_API_PORT="$2"; shift 2 ;;
    --mode) BAGEL_MODE="$2"; shift 2 ;;
    --model_path) BAGEL_MODEL_PATH="$2"; shift 2 ;;
    --python) BAGEL_PYTHON="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${DO_CHECK}" -eq 1 ]]; then
  curl -fsS "http://${BAGEL_API_HOST}:${BAGEL_API_PORT}/health" | python3 -m json.tool
  exit $?
fi

if [[ ! -x "${BAGEL_PYTHON}" && ! -f "${BAGEL_PYTHON}" ]]; then
  echo "Python 不存在: ${BAGEL_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${BAGEL_DIR}/serve_api.py" ]]; then
  echo "未找到 ${BAGEL_DIR}/serve_api.py" >&2
  exit 1
fi
if [[ ! -d "${BAGEL_MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${BAGEL_MODEL_PATH}" >&2
  exit 1
fi

# 与 viewer 内 Bagel 子进程一致：避免 ROS/Qwen PYTHONPATH 污染
unset PYTHONPATH PYTHONHOME || true
export PYTHONPATH="${BAGEL_DIR}"
# 清掉常见代理，避免本机回环被劫持
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.cache/bagel_tmp}"
mkdir -p "${TMPDIR}"
export PYTHONDONTWRITEBYTECODE=1
export BITSANDBYTES_NOWELCOME=1
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:MatMul8bitLt:UserWarning"

echo "[run_bagel_api] python=${BAGEL_PYTHON}"
echo "[run_bagel_api] model=${BAGEL_MODEL_PATH}"
echo "[run_bagel_api] listen=http://${BAGEL_API_HOST}:${BAGEL_API_PORT}  mode=${BAGEL_MODE}"

cd "${BAGEL_DIR}"
exec "${BAGEL_PYTHON}" serve_api.py \
  --host "${BAGEL_API_HOST}" \
  --port "${BAGEL_API_PORT}" \
  --model_path "${BAGEL_MODEL_PATH}" \
  --mode "${BAGEL_MODE}" \
  "${EXTRA_ARGS[@]}"
