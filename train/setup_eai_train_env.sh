#!/usr/bin/env bash
# Create / refresh the dedicated conda env for eai/train LoRA SFT.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3}"
ENV_NAME="${ENV_NAME:-eai-train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu118}"  # driver 535: cu118; cu121 has no torch>=2.6
TORCH_VER="${TORCH_VER:-2.6.0}"
TV_VER="${TV_VER:-0.21.0}"
TA_VER="${TA_VER:-2.6.0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${ROOT}/requirements-train.txt"
CONDA="${CONDA_ROOT}/bin/conda"
PIP="${CONDA_ROOT}/envs/${ENV_NAME}/bin/pip"
PYTHON="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"

if [[ ! -x "${CONDA}" ]]; then
  echo "[error] conda not found: ${CONDA}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "[info] creating conda env ${ENV_NAME} (python=${PYTHON_VERSION})"
  "${CONDA}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "[info] installing torch ${TORCH_VER}+${TORCH_CUDA}"
"${PIP}" install -U pip setuptools wheel
"${PIP}" install \
  "torch==${TORCH_VER}+${TORCH_CUDA}" \
  "torchvision==${TV_VER}+${TORCH_CUDA}" \
  "torchaudio==${TA_VER}+${TORCH_CUDA}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[info] installing HF train stack from ${REQ}"
"${PIP}" install -r "${REQ}"

echo "[info] verifying"
"${PYTHON}" - <<'PY'
import torch, transformers, peft, accelerate
from transformers.utils.import_utils import check_torch_load_is_safe
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__, "accelerate", accelerate.__version__)
check_torch_load_is_safe()
print("check_torch_load_is_safe: OK")
print("python", __import__("sys").executable)
PY

echo "[ok] env ready: ${PYTHON}"
echo "     export PYTHON=${PYTHON}"
echo "     # or: conda activate ${ENV_NAME}"
