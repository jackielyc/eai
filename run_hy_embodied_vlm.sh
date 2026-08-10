#!/usr/bin/env bash
# Hy-Embodied-VLM-1.0 (vLLM OpenAI 服务)
#
# 官方:
#   https://github.com/Tencent-Hunyuan/HY-Embodied
#   https://huggingface.co/tencent/Hy-Embodied-VLM-1.0
#
# 与 show_camera_topics.py:
#   1. 本脚本启动 vLLM serve (默认 http://127.0.0.1:8080/v1)
#   2. 对话面板选预设「Hy-Embodied-VLM-1.0」，勾选「附带相机图」
#
# 硬件: 推荐 4×80GB GPU (TP=4)，全量 BF16 ~86GB
#
# 用法:
#   bash run_hy_embodied_vlm.sh --clone     # 克隆官方仓库到 third_party/
#   bash run_hy_embodied_vlm.sh --install   # 安装 vLLM + 插件 (需 uv)
#   bash run_hy_embodied_vlm.sh             # 启动服务
#   bash run_hy_embodied_vlm.sh --check
#   TP=2 PORT=8080 MODEL_PATH=/path/to/weights bash run_hy_embodied_vlm.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${HY_EMBODIED_REPO:-${SCRIPT_DIR}/third_party/HY-Embodied}"
PORT="${PORT:-${HY_EMBODIED_VLM_PORT:-8080}}"
HOST="${HOST:-${HY_EMBODIED_VLM_HOST:-127.0.0.1}}"
TP="${TP:-4}"
MODEL_PATH="${MODEL_PATH:-tencent/Hy-Embodied-VLM-1.0}"
SERVED_NAME="${SERVED_NAME:-hy_a3b}"

DO_CLONE=0
DO_INSTALL=0
DO_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone) DO_CLONE=1; shift ;;
    --install) DO_INSTALL=1; shift ;;
    --check) DO_CHECK=1; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ "$DO_CHECK" == "1" ]]; then
  if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "OK: Hy-Embodied-VLM 在线 http://${HOST}:${PORT}/v1"
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
    git clone --depth 1 https://github.com/Tencent-Hunyuan/HY-Embodied.git "$REPO_DIR"
  fi
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "未找到仓库: $REPO_DIR"
  echo "请先: bash run_hy_embodied_vlm.sh --clone"
  exit 1
fi

PLUGIN_DIR="${REPO_DIR}/Hy-Embodied-VLM-1.0/inference/vllm"
SERVE_SH="${PLUGIN_DIR}/serve.sh"

if [[ "$DO_INSTALL" == "1" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "需要 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  echo ">>> 安装 vllm==0.14.1 + 插件"
  uv pip install "vllm==0.14.1" --torch-backend auto
  uv pip install -e "$PLUGIN_DIR"
  echo "安装完成"
  exit 0
fi

if [[ ! -f "$SERVE_SH" ]]; then
  echo "未找到 $SERVE_SH — 请 --clone 后重试"
  exit 1
fi

echo ">>> 启动 Hy-Embodied-VLM-1.0"
echo "    MODEL_PATH=$MODEL_PATH"
echo "    TP=$TP  PORT=$PORT  SERVED_NAME=$SERVED_NAME"
echo "    API: http://${HOST}:${PORT}/v1   model=$SERVED_NAME"
echo "    对话面板选: Hy-Embodied-VLM-1.0"

export MODEL_PATH TP PORT SERVED_NAME
# serve.sh 默认绑定；若支持 HOST 则透传
export HOST
cd "$PLUGIN_DIR"
# 部分 serve.sh 仅用 PORT；用 env 覆盖
bash serve.sh
