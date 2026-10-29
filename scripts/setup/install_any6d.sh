#!/usr/bin/env bash
# 安装 Any6D 运行环境（conda 名: Any6D）
# 用法: bash scripts/setup/install_any6d.sh [--skip-weights] [--skip-instantmesh]
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ANY6D_ROOT="${ANY6D_ROOT:-${SCRIPT_DIR}/Any6D}"
MINI="${MINICONDA_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3}"
ENV_NAME="${ANY6D_CONDA_ENV:-Any6D}"
NVDIFFRAST_SRC="${NVDIFFRAST_SRC:-${SCRIPT_DIR}/third_party/nvdiffrast}"
SKIP_WEIGHTS=0
SKIP_IM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --skip-instantmesh) SKIP_IM=1; shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

if [[ ! -d "${ANY6D_ROOT}" ]]; then
  echo "缺少 Any6D 源码: ${ANY6D_ROOT}"
  exit 1
fi

# shellcheck disable=SC1091
source "${MINI}/etc/profile.d/conda.sh"

echo "=== [1/7] conda env ${ENV_NAME} (python 3.9) ==="
# 强制建在 miniconda3/envs（避免 ~/.condarc 的 envs_dirs 指到 RoboDojo_cache）
ENV_DIR="${MINI}/envs/${ENV_NAME}"
if [[ -L "${ENV_DIR}" ]]; then
  echo "警告: ${ENV_DIR} 是软链，请先改为实目录"
  exit 1
fi
if [[ -x "${ENV_DIR}/bin/python" ]]; then
  echo "环境已存在，跳过 create: ${ENV_DIR}"
else
  echo "创建环境: ${ENV_DIR}"
  conda create -y --prefix "${ENV_DIR}" python=3.9
fi
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
export PATH="${ENV_DIR}/bin:${PATH}"
echo "Python=$(which python) $(python -V)"
echo "CONDA_PREFIX=${CONDA_PREFIX:-${ENV_DIR}}"

echo "=== [2/7] eigen + CMAKE_PREFIX_PATH ==="
conda install -y -c conda-forge eigen=3.4.0 pybind11 cmake ninja
EIGEN_PREFIX="$(python -c 'import os,sys; print(os.environ.get("CONDA_PREFIX") or sys.prefix)')"
PY_TAG="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
export CMAKE_PREFIX_PATH="${EIGEN_PREFIX}:${EIGEN_PREFIX}/lib/${PY_TAG}/site-packages/pybind11/share/cmake/pybind11:${CMAKE_PREFIX_PATH:-}"
echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"

echo "=== [3/7] pip requirements (torch cu121 + deps) ==="
cd "${ANY6D_ROOT}"
# 先装 torch，再装其余（避免解析冲突）
python -m pip install -U pip setuptools wheel
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1+cu121 torchvision==0.19.1 torchaudio==2.4.1
# requirements 里已含 torch；用 --no-deps 会漏依赖，改为允许已满足的 torch
python -m pip install -r requirements.txt || {
  echo "WARNING: requirements.txt 部分包失败，继续装核心依赖"
  python -m pip install numpy==1.26.4 opencv-python==4.9.0.80 open3d==0.18.0 \
    trimesh==4.2.2 omegaconf==2.3.0 scipy scikit-image imageio pyyaml \
    transformations roma kornia einops h5py numba ninja pybind11 \
    fvcore iopath || true
}

echo "=== [4/7] nvdiffrast ==="
if [[ -d "${NVDIFFRAST_SRC}" ]]; then
  python -m pip install --no-cache-dir -e "${NVDIFFRAST_SRC}"
else
  python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
fi

echo "=== [5/7] kaolin + pytorch3d ==="
python -m pip install --no-cache-dir kaolin==0.16.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html || \
  echo "WARNING: kaolin 安装失败（InstantMesh 可能不可用，worker 深度凸包仍可用）"
python -m pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
  "pytorch3d==0.7.8+pt2.4.1cu121" || \
  echo "WARNING: pytorch3d wheel 失败，尝试源码跳过"

echo "=== [6/7] build foundationpose extensions ==="
cd "${ANY6D_ROOT}"
CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" bash foundationpose/build_all_conda.sh

echo "=== [7/7] sam2 / instantmesh / bop_toolkit ==="
if [[ -d "${ANY6D_ROOT}/sam2" ]]; then
  (cd "${ANY6D_ROOT}/sam2" && python -m pip install -e .) || echo "WARNING: sam2 editable install failed"
fi
if [[ "${SKIP_IM}" -eq 0 && -d "${ANY6D_ROOT}/instantmesh" ]]; then
  (cd "${ANY6D_ROOT}/instantmesh" && python -m pip install -e .) || echo "WARNING: instantmesh install failed"
fi
if [[ -d "${ANY6D_ROOT}/bop_toolkit" ]]; then
  (cd "${ANY6D_ROOT}/bop_toolkit" && python setup.py install) || \
  (cd "${ANY6D_ROOT}/bop_toolkit" && python -m pip install -e .) || \
  echo "WARNING: bop_toolkit install failed"
fi

if [[ "${SKIP_WEIGHTS}" -eq 0 ]]; then
  echo "=== 下载权重（SAM2 + 尝试 HF InstantMesh；FP 权重需手动/ gdown）==="
  mkdir -p "${ANY6D_ROOT}/sam2/checkpoints" \
           "${ANY6D_ROOT}/foundationpose/weights" \
           "${ANY6D_ROOT}/instantmesh/ckpts"
  SAM2_CKPT="${ANY6D_ROOT}/sam2/checkpoints/sam2.1_hiera_large.pt"
  if [[ ! -f "${SAM2_CKPT}" ]]; then
    echo "下载 SAM2..."
    curl -L --fail --retry 3 -o "${SAM2_CKPT}" \
      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
      || echo "WARNING: SAM2 下载失败"
  else
    echo "SAM2 已存在"
  fi
  # InstantMesh 大文件：仅在缺失时用 huggingface-cli / wget
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download TencentARC/InstantMesh \
      diffusion_pytorch_model.bin instant_mesh_large.ckpt \
      --local-dir "${ANY6D_ROOT}/instantmesh/ckpts" || true
  fi
  FP_MODELS="${FOUNDATIONPOSE_WEIGHTS:-/share_data/projects/mahjong/share/personal/liyichao/models/FoundationPose/weights}"
  mkdir -p "${FP_MODELS}"
  if [[ ! -f "${FP_MODELS}/2024-01-11-20-02-45/model_best.pth" ]]; then
    echo "下载 FoundationPose 权重 → ${FP_MODELS} (HF: gpue/foundationpose-weights)"
    python - <<PY || echo "WARNING: FP 权重下载失败，可手动: huggingface-cli download gpue/foundationpose-weights --local-dir ${FP_MODELS}"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="gpue/foundationpose-weights", local_dir="${FP_MODELS}")
print("OK")
PY
  else
    echo "FoundationPose 权重已存在: ${FP_MODELS}"
  fi
  A6_W="${ANY6D_ROOT}/foundationpose/weights"
  if [[ ! -e "${A6_W}" ]]; then
    ln -s "${FP_MODELS}" "${A6_W}" && echo "linked: ${A6_W} -> ${FP_MODELS}"
  fi
fi

echo "=== 验证 ==="
python - <<'PY'
import importlib
mods = ["torch", "cv2", "trimesh", "open3d", "nvdiffrast.torch", "omegaconf"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK  {m}")
    except Exception as e:
        print(f"FAIL {m}: {e}")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "完成。启动: bash ${SCRIPT_DIR}/scripts/launch/run_any6d.sh"
echo "Python: $(which python)"
