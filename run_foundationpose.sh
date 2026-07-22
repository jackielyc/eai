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
#   bash run_foundationpose.sh --install-gpu
#   bash run_foundationpose.sh --install-gpu --nvdiffrast-src /path/to/nvdiffrast
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
#   NVDIFFRAST_SRC          本地 nvdiffrast 源码目录（GitHub 不可达时用）
#   PYTORCH3D_SRC           本地 pytorch3d 源码目录
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/foundationpose_worker.py"

FP_CONDA_ENV="${FP_CONDA_ENV:-foundationpose}"
FP_PYTHON_VER="${FP_PYTHON_VER:-3.10}"
FP_PORT="${FP_PORT:-8766}"
FP_HOST="${FP_HOST:-127.0.0.1}"
FP_MESH="${FP_MESH:-}"
NVDIFFRAST_SRC="${NVDIFFRAST_SRC:-}"
PYTORCH3D_SRC="${PYTORCH3D_SRC:-}"
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
    sed -n '2,35p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) DO_INSTALL=1; shift ;;
        --install-gpu) DO_INSTALL=1; DO_INSTALL_GPU=1; shift ;;
        --nvdiffrast-src) NVDIFFRAST_SRC="$2"; shift 2 ;;
        --pytorch3d-src) PYTORCH3D_SRC="$2"; shift 2 ;;
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
    "${py}" -m pip install -U ninja packaging

    echo ">>> [2/5] 安装 conda CUDA nvcc / 开发头文件..."
    if ! command -v nvcc >/dev/null 2>&1; then
        conda install -y -c conda-forge "cuda-nvcc=12.4.131" "cuda-cudart-dev=12.4.127" "gxx_linux-64=12"
    fi
    # cusparse.h 等不在 cuda-nvcc 包里；尽量装 conda 版，失败则用 torch 自带的 nvidia/*/include
    conda install -y -c nvidia \
        "cuda-cusparse-dev=12.4.*" \
        "cuda-cublas-dev=12.4.*" \
        "cuda-cudart-dev=12.4.*" 2>/dev/null \
        || echo ">>> conda cuda-*-dev 可选安装跳过（将使用 torch 自带 headers）"

    export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    # RTX 4090 = sm_89；避免编译全部架构
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

    # 把 PyTorch 自带的 CUDA 库头文件加入搜索路径（解决 cusparse.h not found）
    local nvidia_root
    nvidia_root="$("${py}" - <<'PY'
import os, site
for sp in site.getsitepackages():
    root = os.path.join(sp, "nvidia")
    if os.path.isdir(root):
        print(root)
        break
PY
)"
    if [[ -n "${nvidia_root}" && -d "${nvidia_root}" ]]; then
        local inc_paths=()
        local lib_paths=()
        local d
        for d in cusparse cublas cudnn cufft curand cusolver cuda_runtime cuda_nvrtc nccl nvtx; do
            [[ -d "${nvidia_root}/${d}/include" ]] && inc_paths+=("${nvidia_root}/${d}/include")
            [[ -d "${nvidia_root}/${d}/lib" ]] && lib_paths+=("${nvidia_root}/${d}/lib")
        done
        # conda targets include
        [[ -d "${CONDA_PREFIX}/targets/x86_64-linux/include" ]] && \
            inc_paths+=("${CONDA_PREFIX}/targets/x86_64-linux/include")
        [[ -d "${CONDA_PREFIX}/include" ]] && inc_paths+=("${CONDA_PREFIX}/include")

        local IFS=':'
        export CPATH="${inc_paths[*]}${CPATH:+:${CPATH}}"
        export CPLUS_INCLUDE_PATH="${inc_paths[*]}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
        export LIBRARY_PATH="${lib_paths[*]}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
        export LD_LIBRARY_PATH="${lib_paths[*]}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        echo ">>> 已加入 PyTorch nvidia headers: ${nvidia_root}"
    fi

    echo ">>> CUDA_HOME=${CUDA_HOME}"
    echo ">>> TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    echo ">>> CPATH=${CPATH}"
    nvcc --version || echo "警告: nvcc 仍不可用，nvdiffrast 编译可能失败"
    if ! ls "${nvidia_root}/cusparse/include/cusparse.h" >/dev/null 2>&1 \
        && ! ls "${CONDA_PREFIX}"/include/cusparse.h >/dev/null 2>&1 \
        && ! ls "${CONDA_PREFIX}"/targets/x86_64-linux/include/cusparse.h >/dev/null 2>&1; then
        echo "错误: 仍找不到 cusparse.h。请确认已安装 torch（会带 nvidia-cusparse-cu12）" >&2
        exit 1
    fi

    echo ">>> [3/5] 编译安装 nvdiffrast..."
    if [[ -n "${NVDIFFRAST_SRC}" && -d "${NVDIFFRAST_SRC}" ]]; then
        echo ">>> 使用本地源码: ${NVDIFFRAST_SRC}"
        "${py}" -m pip install --no-build-isolation --no-cache-dir "${NVDIFFRAST_SRC}"
    elif [[ -d "${SCRIPT_DIR}/third_party/nvdiffrast" ]]; then
        echo ">>> 使用 third_party/nvdiffrast"
        "${py}" -m pip install --no-build-isolation --no-cache-dir "${SCRIPT_DIR}/third_party/nvdiffrast"
    else
        if ! "${py}" -m pip install --no-build-isolation --no-cache-dir "git+https://github.com/NVlabs/nvdiffrast.git"; then
            echo "" >&2
            echo "错误: 无法从 GitHub 拉取 nvdiffrast（网络受限）。" >&2
            echo "请在可访问 GitHub 的机器上下载后传到本机，再执行:" >&2
            echo "  git clone https://github.com/NVlabs/nvdiffrast.git ${SCRIPT_DIR}/third_party/nvdiffrast" >&2
            echo "  bash ${SCRIPT_DIR}/run_foundationpose.sh --install-gpu --nvdiffrast-src ${SCRIPT_DIR}/third_party/nvdiffrast" >&2
            exit 1
        fi
    fi

    echo ">>> [4/5] 编译安装 pytorch3d..."
    if [[ -n "${PYTORCH3D_SRC}" && -d "${PYTORCH3D_SRC}" ]]; then
        echo ">>> 使用本地源码: ${PYTORCH3D_SRC}"
        "${py}" -m pip install --no-build-isolation "${PYTORCH3D_SRC}"
    elif [[ -d "${SCRIPT_DIR}/third_party/pytorch3d" ]]; then
        echo ">>> 使用 third_party/pytorch3d"
        "${py}" -m pip install --no-build-isolation "${SCRIPT_DIR}/third_party/pytorch3d"
    else
        if ! "${py}" -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"; then
            echo "" >&2
            echo "错误: 无法从 GitHub 拉取 pytorch3d。" >&2
            echo "请下载到 ${SCRIPT_DIR}/third_party/pytorch3d 后重试 --install-gpu" >&2
            exit 1
        fi
    fi

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
    echo ">>> 验证:"
    echo ">>>   ${py} -c \"import nvdiffrast.torch; import pytorch3d; import torch; print('cuda', torch.cuda.is_available())\""
    if ! "${py}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo ""
        echo "警告: 当前进程看不到 GPU（torch.cuda.is_available()=False）。" >&2
        echo "请在宿主机确认: nvidia-smi 正常，且 /dev/nvidia* 存在。" >&2
    fi
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

if ! "${PYTHON}" -c "import nvdiffrast.torch" 2>/dev/null; then
    echo "错误: ${PYTHON} 缺少 nvdiffrast（FoundationPose 渲染必需）" >&2
    echo "请运行: bash ${SCRIPT_DIR}/run_foundationpose.sh --install-gpu" >&2
    echo "完成后重启: bash ${SCRIPT_DIR}/run_foundationpose.sh --mesh \"${FP_MESH:-<object.obj>}\"" >&2
    exit 1
fi

if ! "${PYTHON}" -c "import pytorch3d" 2>/dev/null; then
    echo "错误: ${PYTHON} 缺少 pytorch3d" >&2
    echo "请运行: bash ${SCRIPT_DIR}/run_foundationpose.sh --install-gpu" >&2
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
