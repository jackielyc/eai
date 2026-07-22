#!/usr/bin/env bash
# 启动 CAD / Mesh 生成 Web 界面
#
# 用法:
#   bash run_cad_ui.sh
#   bash run_cad_ui.sh --install-deps
#   CAD_UI_PORT=7861 bash run_cad_ui.sh
#
# 环境变量:
#   MESH_PYTHON     Python（默认优先 foundationpose conda）
#   CAD_UI_HOST     监听地址（默认 0.0.0.0）
#   CAD_UI_PORT     端口（默认 7860）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${SCRIPT_DIR}/requirements-mesh.txt"
UI="${SCRIPT_DIR}/cad_generate_ui.py"
CAD_UI_HOST="${CAD_UI_HOST:-0.0.0.0}"
CAD_UI_PORT="${CAD_UI_PORT:-7860}"
FP_CONDA_ENV="${FP_CONDA_ENV:-foundationpose}"
DO_INSTALL=0
EXTRA=()

pick_python() {
    if [[ -n "${MESH_PYTHON:-}" ]]; then
        echo "${MESH_PYTHON}"
        return
    fi
    local fp_py="/home/psibot/miniconda3/envs/${FP_CONDA_ENV}/bin/python"
    if [[ -x "${fp_py}" ]]; then
        echo "${fp_py}"
        return
    fi
    echo "python3"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-deps) DO_INSTALL=1; shift ;;
        --host) CAD_UI_HOST="$2"; shift 2 ;;
        --port) CAD_UI_PORT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,14p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

MESH_PYTHON="$(pick_python)"

if [[ "${DO_INSTALL}" -eq 1 ]]; then
    echo ">>> 使用 Python: ${MESH_PYTHON}"
    "${MESH_PYTHON}" -m pip install -U pip
    # 去掉可能装不上的 gradio，只装重建依赖
    "${MESH_PYTHON}" -m pip install 'trimesh>=4.0.0' 'open3d>=0.17.0' 'numpy>=1.23.5,<2.0.0' 'matplotlib>=3.7.0'
    if ! command -v colmap >/dev/null 2>&1; then
        echo ">>> 未检测到 COLMAP。照片重建需要: sudo apt install -y colmap"
        echo ">>> 或: bash run_mesh_reconstruct.sh --install-deps"
    else
        echo ">>> COLMAP: ok"
    fi
    exit 0
fi

if ! "${MESH_PYTHON}" -c "import trimesh, open3d, matplotlib" >/dev/null 2>&1; then
    echo ">>> 缺少依赖，正在安装 …"
    "${MESH_PYTHON}" -m pip install 'trimesh>=4.0.0' 'open3d>=0.17.0' 'numpy>=1.23.5,<2.0.0' 'matplotlib>=3.7.0'
fi

echo ">>> Python: ${MESH_PYTHON}"
echo ">>> 打开浏览器: http://127.0.0.1:${CAD_UI_PORT}"
echo ">>> 远端访问:   http://<本机IP>:${CAD_UI_PORT}"
exec "${MESH_PYTHON}" "${UI}" --host "${CAD_UI_HOST}" --port "${CAD_UI_PORT}" "${EXTRA[@]}"
