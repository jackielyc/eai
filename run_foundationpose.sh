#!/usr/bin/env bash
# FoundationPose 6D 位姿 HTTP 服务
#
# 与 show_camera_topics.py 配合:
#   1. 宿主机启动本服务
#   2. viewer「分割」标签 → 位姿后端选 FoundationPose，勾选 HTTP
#   3. 指定物体 mesh (.obj) 路径
#
# 前置: 克隆并安装 NVlabs/FoundationPose
#   git clone https://github.com/NVlabs/FoundationPose.git
#   # 按官方 README 安装依赖 (CUDA, nvdiffrast, pytorch3d 等)
#   export FOUNDATIONPOSE_ROOT=~/FoundationPose
#
# 用法:
#   bash run_foundationpose.sh --install
#   bash run_foundationpose.sh --mesh /path/to/object.obj
#   bash run_foundationpose.sh --check
#   bash run_foundationpose.sh --host 0.0.0.0
#
# 无 CAD mesh（方案 A）:
#   bash run_mesh_reconstruct.sh --install-deps
#   bash run_mesh_reconstruct.sh --images ~/photos/object --name my_obj
#   bash run_foundationpose.sh --mesh meshes/my_obj/reconstructed.obj
#
# 环境变量:
#   FP_CONDA_ENV            Conda 环境名 (默认 foundationpose)
#   FP_PORT                 端口 (默认 8766)
#   FP_HOST                 监听地址 (默认 127.0.0.1)
#   FP_MESH                 默认 mesh 路径
#   FOUNDATIONPOSE_ROOT     FoundationPose 源码根目录
#   FP_PYTHON               指定 Python (覆盖 conda)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/foundationpose_worker.py"

FP_CONDA_ENV="${FP_CONDA_ENV:-foundationpose}"
FP_PYTHON_VER="${FP_PYTHON_VER:-3.10}"
FP_PORT="${FP_PORT:-8766}"
FP_HOST="${FP_HOST:-127.0.0.1}"
FP_MESH="${FP_MESH:-}"
if [[ -z "${FOUNDATIONPOSE_ROOT:-}" ]]; then
    if [[ -d "${SCRIPT_DIR}/FoundationPose" ]]; then
        FOUNDATIONPOSE_ROOT="${SCRIPT_DIR}/FoundationPose"
    else
        FOUNDATIONPOSE_ROOT="${HOME}/FoundationPose"
    fi
fi
export FOUNDATIONPOSE_ROOT

DO_INSTALL=0
DO_INSTALL_GPU=0
DO_CHECK=0
EXTRA_ARGS=()

usage() {
    sed -n '2,28p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) DO_INSTALL=1; shift ;;
        --install-gpu) DO_INSTALL=1; DO_INSTALL_GPU=1; shift ;;
        --check) DO_CHECK=1; shift ;;
        --host) FP_HOST="$2"; shift 2 ;;
        --port) FP_PORT="$2"; shift 2 ;;
        --mesh) FP_MESH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

init_conda() {
    if [[ -n "${FP_PYTHON:-}" ]]; then
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        return 0
    fi
    for base in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda"; do
        if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "${base}/etc/profile.d/conda.sh"
            return 0
        fi
    done
    echo "错误: 未找到 conda，请先安装 Miniconda 并 conda init" >&2
    exit 1
}

conda_env_python() {
    echo "$(conda info --base)/envs/${FP_CONDA_ENV}/bin/python"
}

conda_env_exists() {
    init_conda
    conda env list | awk '{print $1}' | grep -qx "${FP_CONDA_ENV}"
}

pick_python() {
    if [[ -n "${FP_PYTHON:-}" ]]; then
        echo "${FP_PYTHON}"
        return
    fi
    init_conda
    local py
    py="$(conda_env_python)"
    if [[ -x "${py}" ]]; then
        echo "${py}"
        return
    fi
    echo "错误: Conda 环境 ${FP_CONDA_ENV} 不存在，请先: bash run_foundationpose.sh --install" >&2
    exit 1
}

install_fp_env() {
    init_conda
    echo ">>> 创建 Conda 环境: ${FP_CONDA_ENV} (Python ${FP_PYTHON_VER})"
    if conda_env_exists; then
        echo ">>> 环境已存在，跳过 conda create"
    else
        conda create -n "${FP_CONDA_ENV}" "python=${FP_PYTHON_VER}" -y
    fi
    # shellcheck disable=SC1091
    conda activate "${FP_CONDA_ENV}"
    pip install -U pip wheel
    pip install numpy opencv-python-headless trimesh scipy scikit-learn PyYAML imageio
    pip install -r "${SCRIPT_DIR}/requirements-mesh.txt" 2>/dev/null || pip install open3d
    echo ""
    echo ">>> 基础依赖已安装（trimesh / open3d）。"
    if [[ "${DO_INSTALL_GPU}" -eq 1 ]]; then
        install_fp_gpu_deps
    else
        echo ">>> 安装 GPU 推理栈（torch + nvdiffrast + pytorch3d）:"
        echo ">>>   bash ${SCRIPT_DIR}/run_foundationpose.sh --install-gpu"
    fi
}

install_fp_gpu_deps() {
    # shellcheck disable=SC1091
    conda activate "${FP_CONDA_ENV}"
    local py="${CONDA_PREFIX}/bin/python"
    local fp_req="${FOUNDATIONPOSE_ROOT}/requirements.txt"

    echo ">>> [1/5] 安装 PyTorch (cu124，适配 RTX 4090 等)..."
    "${py}" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    echo ">>> [2/5] 安装 conda CUDA nvcc（系统无 /usr/local/cuda 时必需）..."
    if ! command -v nvcc >/dev/null 2>&1; then
        conda install -y -c conda-forge "cuda-nvcc=12.4.131" "cuda-cudart-dev=12.4.127" "gxx_linux-64=12"
    fi

    export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    echo ">>> CUDA_HOME=${CUDA_HOME}"
    nvcc --version || echo "警告: nvcc 仍不可用，nvdiffrast 编译可能失败"

    echo ">>> [3/5] 编译安装 nvdiffrast..."
    "${py}" -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

    echo ">>> [4/5] 编译安装 pytorch3d..."
    "${py}" -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

    if [[ -f "${fp_req}" ]]; then
        echo ">>> [5/5] 安装 FoundationPose requirements.txt..."
        "${py}" -m pip install -r "${fp_req}"
    fi

    if [[ -f "${FOUNDATIONPOSE_ROOT}/build_all_conda.sh" ]]; then
        echo ">>> 编译 mycpp 扩展..."
        (cd "${FOUNDATIONPOSE_ROOT}" && bash build_all_conda.sh)
    fi

    echo ""
    echo ">>> GPU 依赖安装完成。请确认 weights/ 已下载:"
    echo ">>>   ${FOUNDATIONPOSE_ROOT}/weights/2023-10-28-18-33-37"
    echo ">>>   ${FOUNDATIONPOSE_ROOT}/weights/2024-01-11-20-02-45"
    echo ">>> 验证: ${py} -c \"import nvdiffrast.torch; import torch; print(torch.cuda.is_available())\""
}

check_health() {
    local url="http://${FP_HOST}:${FP_PORT}/health"
    if [[ "${FP_HOST}" == "0.0.0.0" ]]; then
        url="http://127.0.0.1:${FP_PORT}/health"
    fi
    echo ">>> GET ${url}"
    if curl -sf "${url}"; then
        echo ""
        echo ">>> FoundationPose 服务在线"
        return 0
    fi
    echo ">>> FoundationPose 服务未响应" >&2
    return 1
}

run_worker() {
    local python_exe="$1"
    shift
    export FOUNDATIONPOSE_ROOT
    exec "${python_exe}" "${WORKER}" \
        --serve \
        --host "${FP_HOST}" \
        --port "${FP_PORT}" \
        ${FP_MESH:+--mesh "${FP_MESH}"} \
        "$@"
}

if [[ ! -f "${WORKER}" ]]; then
    echo "错误: 未找到 ${WORKER}" >&2
    exit 1
fi

if [[ "${DO_INSTALL}" -eq 1 ]]; then
    install_fp_env
    exit 0
fi

PYTHON="$(pick_python)"

if [[ "${DO_CHECK}" -eq 1 ]]; then
    check_health
    exit $?
fi

if [[ ! -d "${FOUNDATIONPOSE_ROOT}" ]]; then
    echo "警告: FOUNDATIONPOSE_ROOT 不存在: ${FOUNDATIONPOSE_ROOT}" >&2
    echo "请 clone: git clone https://github.com/NVlabs/FoundationPose.git ${FOUNDATIONPOSE_ROOT}" >&2
fi

if ! "${PYTHON}" - <<'PY' 2>/dev/null; then
import cv2  # noqa: F401
import numpy  # noqa: F401
PY
    echo "错误: ${PYTHON} 缺少基础依赖，请先: bash run_foundationpose.sh --install" >&2
    exit 1
fi

VIEWER_URL="http://127.0.0.1:${FP_PORT}"
echo ">>> Python: ${PYTHON}"
echo ">>> FP root: ${FOUNDATIONPOSE_ROOT}"
echo ">>> Mesh: ${FP_MESH:-(请求时指定)}"
echo ">>> Listen: ${FP_HOST}:${FP_PORT}"
echo ""
echo ">>> viewer 中: 分割 → 位姿 FoundationPose → 勾选 FP HTTP"
echo ">>> export FP_USE_HTTP=1"
echo ">>> export FP_SERVER_URL=${VIEWER_URL}"
echo ""

run_worker "${PYTHON}" "${EXTRA_ARGS[@]}"
