#!/usr/bin/env bash
# 方案 A：多视角照片 / Meshroom mesh → FoundationPose 可用 .obj
#
# 依赖:
#   - COLMAP（照片重建）: sudo apt install -y colmap
#   - Python: pip install -r requirements-mesh.txt
#
# 用法:
#   # 安装系统与 Python 依赖
#   bash run_mesh_reconstruct.sh --install-deps
#
#   # 从手机/相机多视角照片重建
#   bash run_mesh_reconstruct.sh --images ~/photos/my_cup --name my_cup
#
#   # 从 Meshroom 等已导出的 mesh 后处理（无需 COLMAP）
#   bash run_mesh_reconstruct.sh --import-mesh ~/Downloads/mesh.obj --name my_cup
#
#   # 重建完成后启动 FoundationPose
#   bash run_mesh_reconstruct.sh --images ~/photos/my_cup --name my_cup --start-fp
#
# 输出:
#   meshes/<name>/reconstructed.obj
#   meshes/<name>/manifest.json
#
# 环境变量:
#   MESH_PYTHON          Python 解释器（默认 python3）
#   MESH_TARGET_EXTENT_M 目标最大边长/米（默认 0.15）
#   FP_HOST / FP_PORT    传给 run_foundationpose.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECON="${SCRIPT_DIR}/photogrammetry_reconstruct.py"
REQ="${SCRIPT_DIR}/requirements-mesh.txt"
FP_SCRIPT="${SCRIPT_DIR}/run_foundationpose.sh"

MESH_TARGET_EXTENT_M="${MESH_TARGET_EXTENT_M:-0.15}"
FP_CONDA_ENV="${FP_CONDA_ENV:-foundationpose}"

pick_mesh_python() {
    if [[ -n "${MESH_PYTHON:-}" ]]; then
        echo "${MESH_PYTHON}"
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        local fp_py
        fp_py="$(conda info --base)/envs/${FP_CONDA_ENV}/bin/python"
        if [[ -x "${fp_py}" ]]; then
            echo "${fp_py}"
            return
        fi
    fi
    echo "python3"
}

MESH_PYTHON="$(pick_mesh_python)"
IMAGES_DIR=""
IMPORT_MESH=""
NAME=""
OUT_DIR="${SCRIPT_DIR}/meshes"
START_FP=0
DO_INSTALL=0
EXTRA_PY=()

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-deps) DO_INSTALL=1; shift ;;
        --images) IMAGES_DIR="$2"; shift 2 ;;
        --import-mesh) IMPORT_MESH="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --target-extent-m) MESH_TARGET_EXTENT_M="$2"; shift 2 ;;
        --start-fp) START_FP=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_PY+=("$1"); shift ;;
    esac
done

install_deps() {
    echo ">>> 安装 COLMAP（需 sudo）…"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y colmap
    else
        echo "警告: 未找到 apt-get，请手动安装 COLMAP" >&2
    fi
    echo ">>> 安装 Python 依赖 …"
    "${MESH_PYTHON}" -m pip install -U pip
    "${MESH_PYTHON}" -m pip install -r "${REQ}"
    echo ""
    echo ">>> 完成。拍照建议见: ${SCRIPT_DIR}/meshes/README.md"
    if command -v colmap >/dev/null 2>&1; then
        echo ">>> COLMAP: $(colmap -h 2>&1 | head -1 || true)"
    fi
}

if [[ "${DO_INSTALL}" -eq 1 ]]; then
    install_deps
    exit 0
fi

if [[ ! -f "${RECON}" ]]; then
    echo "错误: 未找到 ${RECON}" >&2
    exit 1
fi

if [[ -z "${NAME}" ]]; then
    echo "错误: 必须指定 --name <物体名>" >&2
    usage
    exit 2
fi

if [[ -n "${IMAGES_DIR}" && -n "${IMPORT_MESH}" ]]; then
    echo "错误: --images 与 --import-mesh 不能同时使用" >&2
    exit 2
fi

if [[ -z "${IMAGES_DIR}" && -z "${IMPORT_MESH}" ]]; then
    echo "错误: 请指定 --images 或 --import-mesh" >&2
    usage
    exit 2
fi

CMD=(
    "${MESH_PYTHON}" "${RECON}"
    --name "${NAME}"
    --out-dir "${OUT_DIR}"
    --target-extent-m "${MESH_TARGET_EXTENT_M}"
)

if [[ -n "${IMAGES_DIR}" ]]; then
    CMD+=(--images "${IMAGES_DIR}")
else
    CMD+=(--import-mesh "${IMPORT_MESH}")
fi

if [[ ${#EXTRA_PY[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_PY[@]}")
fi

echo ">>> ${CMD[*]}"
"${CMD[@]}"
RESULT=$?

if [[ "${RESULT}" -ne 0 ]]; then
    exit "${RESULT}"
fi

MESH_OUT="${OUT_DIR}/${NAME}/reconstructed.obj"
if [[ ! -f "${MESH_OUT}" ]]; then
    echo "错误: 未生成 ${MESH_OUT}" >&2
    exit 1
fi

echo ""
echo ">>> 重建 mesh: ${MESH_OUT}"
echo ">>> 启动 FoundationPose:"
echo ">>>   bash ${FP_SCRIPT} --host 0.0.0.0 --mesh ${MESH_OUT}"
echo ">>> viewer: FP_USE_HTTP=1 FP_SERVER_URL=http://127.0.0.1:8766"
echo ">>>         分割 → FoundationPose → mesh 填: ${MESH_OUT}"

if [[ "${START_FP}" -eq 1 ]]; then
    if [[ ! -x "${FP_SCRIPT}" && ! -f "${FP_SCRIPT}" ]]; then
        echo "错误: 未找到 ${FP_SCRIPT}" >&2
        exit 1
    fi
    echo ""
    echo ">>> 启动 FoundationPose worker …"
    exec bash "${FP_SCRIPT}" --host "${FP_HOST:-0.0.0.0}" --mesh "${MESH_OUT}"
fi
