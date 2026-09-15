#!/usr/bin/env bash
# 为 Bagel tab 准备独立 Python 环境（不要用 ros-humble）。
set -euo pipefail
EAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY310="${BAGEL_BASE_PYTHON:-/home/psibot/miniconda3/envs/Qwen2.5-VL/bin/python}"
VENV="${BAGEL_VENV:-${EAI_DIR}/.cache/bagel_venv}"
PIP_INDEX="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${EAI_DIR}/.cache/pip}"
mkdir -p "$(dirname "$VENV")" "$PIP_CACHE_DIR"

if [[ ! -x "$PY310" ]]; then
  echo "找不到 Python3.10: $PY310" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo ">>> 创建 venv: $VENV"
  "$PY310" -m venv "$VENV"
fi

echo ">>> 安装 PyTorch cu121…"
"$VENV/bin/python" -m pip install -U pip setuptools wheel -i "$PIP_INDEX"
"$VENV/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo ">>> 安装 Bagel / Gradio 依赖…"
"$VENV/bin/python" -m pip install -i "$PIP_INDEX" \
  'gradio' 'accelerate>=0.34.0' 'bitsandbytes' 'einops==0.8.1' \
  'huggingface_hub==0.29.1' 'safetensors==0.4.5' 'transformers==4.49.0' \
  'numpy<2' 'opencv-python-headless' 'PyYAML' 'Requests' \
  'sentencepiece' 'scipy' 'matplotlib' 'Pillow'

"$VENV/bin/python" - <<'PY'
import gradio, torch, transformers, huggingface_hub
print(
    "OK",
    "gradio", gradio.__version__,
    "torch", torch.__version__,
    "transformers", transformers.__version__,
    "hub", huggingface_hub.__version__,
    "cuda", torch.cuda.is_available(),
)
PY

echo ">>> 完成。Bagel tab 的 Python 请填:"
echo "    $VENV/bin/python"
echo "模型默认:"
echo "    ${EAI_DIR}/.cache/bagel_models/BAGEL-7B-MoT"
