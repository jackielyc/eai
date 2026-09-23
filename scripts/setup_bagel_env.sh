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

# Blackwell (sm_120) 需要 PyTorch >=2.7 + CUDA 12.8 wheels；cu121/cu124/cu126 会报
# "no kernel image is available for execution on the device"。
TORCH_INDEX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
echo ">>> 安装 PyTorch (${TORCH_INDEX})…"
"$VENV/bin/python" -m pip install -U pip setuptools wheel -i "$PIP_INDEX"
"$VENV/bin/python" -m pip install torch torchvision \
  --index-url "$TORCH_INDEX"

echo ">>> 安装 Bagel / Gradio 依赖…"
"$VENV/bin/python" -m pip install -i "$PIP_INDEX" \
  'gradio' 'accelerate>=0.34.0' 'bitsandbytes' 'einops==0.8.1' \
  'huggingface_hub==0.29.1' 'safetensors==0.4.5' 'transformers==4.49.0' \
  'numpy<2' 'opencv-python-headless' 'PyYAML' 'Requests' \
  'sentencepiece' 'scipy' 'matplotlib' 'Pillow' 'packaging' 'ninja'

# Bagel modeling 硬依赖 flash_attn；换 torch 后需按当前 CUDA 重编。
if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
  echo ">>> 编译安装 flash-attn（可能较久；SKIP_FLASH_ATTN=1 可跳过）…"
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
  "$VENV/bin/python" -m pip install -i "$PIP_INDEX" flash-attn --no-build-isolation
fi

"$VENV/bin/python" - <<'PY'
import gradio, torch, transformers, huggingface_hub
arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
print(
    "OK",
    "gradio", gradio.__version__,
    "torch", torch.__version__,
    "transformers", transformers.__version__,
    "hub", huggingface_hub.__version__,
    "cuda", torch.cuda.is_available(),
    "torch_cuda", torch.version.cuda,
    "arch", arch,
)
if torch.cuda.is_available() and not any("120" in a or "100" in a for a in arch):
    print("WARN: 当前 PyTorch wheel 可能不含 Blackwell (sm_120)；请用 cu128 重装。")
PY

echo ">>> 完成。Bagel tab 的 Python 请填:"
echo "    $VENV/bin/python"
echo "模型默认:"
echo "    ${EAI_DIR}/.cache/bagel_models/BAGEL-7B-MoT"
