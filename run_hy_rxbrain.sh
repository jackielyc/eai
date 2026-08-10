#!/usr/bin/env bash
# Hy-Embodied-RxBrain-1.0 OpenAI-compatible VQA 服务
#
# 官方:
#   https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0
#   https://huggingface.co/tencent/Hy-Embodied-RxBrain-1.0
#
# 与 show_camera_topics.py:
#   1. 本脚本启动 hy_rxbrain_worker.py (默认 http://127.0.0.1:8090/v1)
#   2. 对话面板选预设「Hy-Embodied-RxBrain-1.0」，勾选「附带相机图」
#
# 用法:
#   bash run_hy_rxbrain.sh --clone
#   bash run_hy_rxbrain.sh --download   # hf download 权重到 weights/
#   bash run_hy_rxbrain.sh --install    # pip install -r requirements.txt
#   bash run_hy_rxbrain.sh
#   bash run_hy_rxbrain.sh --check
#   bash run_hy_rxbrain.sh --host 0.0.0.0   # Docker viewer 访问
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/hy_rxbrain_worker.py"
REPO_DIR="${HY_RXBRAIN_REPO:-${SCRIPT_DIR}/third_party/Hy-Embodied-RxBrain-1.0}"
CKPT_DIR="${HY_RXBRAIN_CKPT:-${SCRIPT_DIR}/weights/Hy-Embodied-RxBrain-1.0}"
PORT="${HY_RXBRAIN_PORT:-8090}"
HOST="${HY_RXBRAIN_HOST:-127.0.0.1}"
PY="${HY_RXBRAIN_PYTHON:-python3}"

DO_CLONE=0
DO_DOWNLOAD=0
DO_INSTALL=0
DO_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone) DO_CLONE=1; shift ;;
    --download) DO_DOWNLOAD=1; shift ;;
    --install) DO_INSTALL=1; shift ;;
    --check) DO_CHECK=1; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --repo) REPO_DIR="$2"; shift 2 ;;
    --ckpt) CKPT_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ "$DO_CHECK" == "1" ]]; then
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "OK: RxBrain 在线 http://${HOST}:${PORT}/v1"
    exit 0
  fi
  echo "离线: http://${HOST}:${PORT}/v1"
  exit 1
fi

if [[ "$DO_CLONE" == "1" ]]; then
  mkdir -p "$(dirname "$REPO_DIR")"
  if [[ -d "$REPO_DIR/.git" ]]; then
    echo "已存在: $REPO_DIR"
  else
    git clone --depth 1 \
      https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0.git \
      "$REPO_DIR"
  fi
fi

if [[ "$DO_DOWNLOAD" == "1" ]]; then
  if ! command -v hf >/dev/null 2>&1 && ! "$PY" -c "import huggingface_hub" 2>/dev/null; then
    "$PY" -m pip install -U "huggingface_hub[cli]"
  fi
  mkdir -p "$CKPT_DIR"
  echo ">>> 下载 tencent/Hy-Embodied-RxBrain-1.0 -> $CKPT_DIR"
  if command -v hf >/dev/null 2>&1; then
    hf download tencent/Hy-Embodied-RxBrain-1.0 --local-dir "$CKPT_DIR"
  else
    "$PY" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("tencent/Hy-Embodied-RxBrain-1.0", local_dir="$CKPT_DIR")
print("done")
PY
  fi
fi

if [[ "$DO_INSTALL" == "1" ]]; then
  if [[ ! -d "$REPO_DIR" ]]; then
    echo "请先 --clone"
    exit 1
  fi
  echo ">>> 安装 pinned transformers + requirements"
  "$PY" -m pip install \
    "git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a"
  if [[ -f "$REPO_DIR/requirements.txt" ]]; then
    "$PY" -m pip install -r "$REPO_DIR/requirements.txt"
  fi
  echo "安装完成（图像生成还需 FLUX VAE ae.safetensors；VQA 不需要）"
  exit 0
fi

if [[ ! -f "$WORKER" ]]; then
  echo "缺少 $WORKER"
  exit 1
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "未找到仓库 $REPO_DIR — 请: bash run_hy_rxbrain.sh --clone"
  exit 1
fi
if [[ ! -f "$CKPT_DIR/config.json" ]]; then
  echo "未找到权重 $CKPT_DIR/config.json — 请: bash run_hy_rxbrain.sh --download"
  exit 1
fi

export HY_RXBRAIN_REPO="$REPO_DIR"
export HY_RXBRAIN_CKPT="$CKPT_DIR"
export HY_RXBRAIN_HOST="$HOST"
export HY_RXBRAIN_PORT="$PORT"

echo ">>> 启动 Hy-Embodied-RxBrain worker"
echo "    repo=$REPO_DIR"
echo "    ckpt=$CKPT_DIR"
echo "    API: http://${HOST}:${PORT}/v1   model=hy-rxbrain"
echo "    对话面板选: Hy-Embodied-RxBrain-1.0 + 附带相机图"

exec "$PY" "$WORKER" --host "$HOST" --port "$PORT" --repo "$REPO_DIR" --ckpt "$CKPT_DIR"
