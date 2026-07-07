#!/usr/bin/env bash
# SAM3 本地分割服务 — 一键安装 / 启动 / 健康检查
#
# 与 show_camera_topics.py 配合:
#   1. 本脚本启动 SAM3 HTTP 服务（宿主机）
#   2. viewer 工具栏选「SAM3 点提示」并勾选 HTTP
#   3. 或在启动 viewer 前:
#        export SAM3_USE_HTTP=1
#        export SAM3_SERVER_URL=http://127.0.0.1:8765
#
# 用法:
#   bash run_sam3.sh --install     # 首次: 创建 venv + 安装依赖
#   bash run_sam3.sh               # 启动服务 (默认 127.0.0.1:8765)
#   bash run_sam3.sh --check       # 检查服务是否在线
#   bash run_sam3.sh --host 0.0.0.0  # 允许 Docker 内 viewer 访问
#
# 环境变量:
#   SAM3_VENV      虚拟环境路径 (默认 ~/.venvs/sam3)
#   SAM3_MODEL     权重路径 (默认 $SCRIPT_DIR/sam3.pt)
#   SAM3_PORT      端口 (默认 8765)
#   SAM3_HOST      监听地址 (默认 127.0.0.1)
#   SAM3_PYTHON    指定 Python (覆盖 venv)
#   TORCH_INDEX_URL  PyTorch 安装源 (默认 cu126)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/sam3_segment_worker.py"
REQ_FILE="${SCRIPT_DIR}/requirements-sam3.txt"

SAM3_VENV="${SAM3_VENV:-${HOME}/.venvs/sam3}"
SAM3_MODEL="${SAM3_MODEL:-${SCRIPT_DIR}/sam3.pt}"
SAM3_PORT="${SAM3_PORT:-8765}"
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

DO_INSTALL=0
DO_CHECK=0
EXTRA_ARGS=()

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    echo ""
    echo "选项:"
    echo "  --install          创建 ${SAM3_VENV} 并安装 requirements-sam3.txt"
    echo "  --check            GET /health，检查服务是否运行"
    echo "  --host ADDR        监听地址 (默认 ${SAM3_HOST})"
    echo "  --port PORT        端口 (默认 ${SAM3_PORT})"
    echo "  --model PATH       sam3.pt 路径 (默认 ${SAM3_MODEL})"
    echo "  -h, --help         显示帮助"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) DO_INSTALL=1; shift ;;
        --check) DO_CHECK=1; shift ;;
        --host) SAM3_HOST="$2"; shift 2 ;;
        --port) SAM3_PORT="$2"; shift 2 ;;
        --model) SAM3_MODEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

pick_python() {
    if [[ -n "${SAM3_PYTHON:-}" ]]; then
        echo "${SAM3_PYTHON}"
        return
    fi
    if [[ -x "${SAM3_VENV}/bin/python" ]]; then
        echo "${SAM3_VENV}/bin/python"
        return
    fi
    if command -v python3.10 >/dev/null 2>&1; then
        echo python3.10
        return
    fi
    echo python3
}

install_sam3_env() {
    echo ">>> 创建虚拟环境: ${SAM3_VENV}"
    mkdir -p "$(dirname "${SAM3_VENV}")"
    if [[ ! -x "${SAM3_VENV}/bin/python" ]]; then
        python3.10 -m venv "${SAM3_VENV}" 2>/dev/null || python3 -m venv "${SAM3_VENV}"
    fi
    # shellcheck disable=SC1091
    source "${SAM3_VENV}/bin/activate"
    pip install -U pip wheel
    echo ">>> 安装 PyTorch (${TORCH_INDEX_URL})..."
    pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
    echo ">>> 安装 SAM3 依赖..."
    pip install -r "${REQ_FILE}"
    echo ""
    echo ">>> 安装完成。请下载 sam3.pt 到: ${SAM3_MODEL}"
    echo ">>> HuggingFace: https://huggingface.co/facebook/sam3 (需申请访问)"
    echo ">>> 然后运行: bash ${SCRIPT_DIR}/run_sam3.sh"
}

check_health() {
    local url="http://${SAM3_HOST}:${SAM3_PORT}/health"
    if [[ "${SAM3_HOST}" == "0.0.0.0" ]]; then
        url="http://127.0.0.1:${SAM3_PORT}/health"
    fi
    echo ">>> GET ${url}"
    if curl -sf "${url}"; then
        echo ""
        echo ">>> SAM3 服务在线"
        return 0
    fi
    echo ">>> SAM3 服务未响应" >&2
    return 1
}

if [[ ! -f "${WORKER}" ]]; then
    echo "错误: 未找到 ${WORKER}" >&2
    exit 1
fi

if [[ "${DO_INSTALL}" -eq 1 ]]; then
    install_sam3_env
    exit 0
fi

PYTHON="$(pick_python)"

if [[ "${DO_CHECK}" -eq 1 ]]; then
    check_health
    exit $?
fi

if [[ ! -f "${SAM3_MODEL}" ]]; then
    echo "错误: 未找到模型权重 ${SAM3_MODEL}" >&2
    echo "" >&2
    echo "请先:" >&2
    echo "  1. 在 HuggingFace 申请 facebook/sam3 并下载 sam3.pt" >&2
    echo "  2. 或: SAM3_MODEL=/path/to/sam3.pt bash run_sam3.sh" >&2
    echo "  3. 首次安装: bash run_sam3.sh --install" >&2
    exit 1
fi

if ! "${PYTHON}" - <<'PY' 2>/dev/null; then
import ultralytics  # noqa: F401
import cv2  # noqa: F401
PY
    echo "错误: ${PYTHON} 缺少 ultralytics，请先: bash run_sam3.sh --install" >&2
    exit 1
fi

SERVER_URL="http://${SAM3_HOST}:${SAM3_PORT}"
if [[ "${SAM3_HOST}" == "0.0.0.0" ]]; then
    VIEWER_URL="http://127.0.0.1:${SAM3_PORT}"
else
    VIEWER_URL="${SERVER_URL}"
fi

echo ">>> Python:  ${PYTHON}"
echo ">>> Model:   ${SAM3_MODEL}"
echo ">>> Listen:  ${SAM3_HOST}:${SAM3_PORT}"
echo ">>> Worker:  ${WORKER}"
echo ""
echo ">>> 启动后在 viewer 中:"
echo ">>>   分割 → SAM3 点提示 → 勾选 HTTP"
echo ">>>   export SAM3_USE_HTTP=1"
echo ">>>   export SAM3_SERVER_URL=${VIEWER_URL}"
echo ""
echo ">>> Docker 内 viewer 访问宿主机 SAM3 时:"
echo ">>>   bash run_sam3.sh --host 0.0.0.0"
echo ">>>   export SAM3_SERVER_URL=http://<宿主机IP>:${SAM3_PORT}"
echo ""

exec "${PYTHON}" "${WORKER}" \
    --serve \
    --host "${SAM3_HOST}" \
    --port "${SAM3_PORT}" \
    --model "${SAM3_MODEL}" \
    "${EXTRA_ARGS[@]}"
