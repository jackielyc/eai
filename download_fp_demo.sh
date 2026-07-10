#!/usr/bin/env bash
# 下载 FoundationPose 官方 demo_data（含 mustard 黄芥末瓶 mesh）
#
# 用法:
#   bash download_fp_demo.sh
#
# 需要能访问 Google Drive；下载后 mesh 路径:
#   eai/FoundationPose/demo_data/mustard0/mesh/textured_simple.obj
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FP_ROOT="${FOUNDATIONPOSE_ROOT:-${SCRIPT_DIR}/FoundationPose}"
OUT_DIR="${FP_ROOT}/demo_data"

if [[ ! -d "${FP_ROOT}" ]]; then
    echo "错误: 未找到 ${FP_ROOT}" >&2
    echo "请先: git clone https://github.com/NVlabs/FoundationPose.git ${FP_ROOT}" >&2
    exit 1
fi

pick_python() {
    if [[ -n "${FP_PYTHON:-}" && -x "${FP_PYTHON}" ]]; then
        echo "${FP_PYTHON}"
        return
    fi
    if [[ -x "${HOME}/miniconda3/envs/foundationpose/bin/python" ]]; then
        echo "${HOME}/miniconda3/envs/foundationpose/bin/python"
        return
    fi
    echo "python3"
}

PYTHON="$(pick_python)"
if ! "${PYTHON}" -c "import gdown" 2>/dev/null; then
    echo ">>> 安装 gdown..."
    "${PYTHON}" -m pip install -U gdown
fi

GDOWN="$("${PYTHON}" -c 'import shutil, gdown; print(shutil.which("gdown") or "")')"
if [[ -z "${GDOWN}" ]]; then
    GDOWN="${PYTHON} -m gdown"
fi

echo ">>> 下载官方 demo_data 到 ${OUT_DIR}"
echo ">>> Google Drive: https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP"
mkdir -p "${OUT_DIR}"
eval "${GDOWN} --folder 'https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP' -O '${OUT_DIR}'"

MESH="${OUT_DIR}/mustard0/mesh/textured_simple.obj"
if [[ -f "${MESH}" ]]; then
    echo ""
    echo ">>> 下载成功: ${MESH}"
    echo ">>> 启动 worker:"
    echo ">>>   bash ${SCRIPT_DIR}/run_foundationpose.sh --host 0.0.0.0 --mesh ${MESH}"
else
    echo ">>> 下载完成但未找到 ${MESH}，请检查 ${OUT_DIR} 目录结构" >&2
    exit 1
fi
