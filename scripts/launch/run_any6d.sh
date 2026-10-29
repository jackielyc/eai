#!/usr/bin/env bash
# Any6D 6D 位姿 HTTP 服务（任意物体 / 可选无 CAD）
#
# 与 compat/show_camera_topics.py 配合:
#   1. 宿主机: bash scripts/launch/run_any6d.sh
#   2. viewer「分割」→ 位姿后端选 Any6D，勾选 HTTP
#   3. mesh 可留空（深度凸包 model-free），或填 .obj
#
# 前置:
#   git clone https://github.com/taeyeopl/Any6D.git eai/Any6D
#   按 Any6D README 装 conda 环境（建议名 Any6D），并准备 foundationpose/weights
#   可软链复用: ln -s ../FoundationPose/weights Any6D/foundationpose/weights
#
# 用法:
#   bash scripts/launch/run_any6d.sh --check
#   bash scripts/launch/run_any6d.sh
#   bash scripts/launch/run_any6d.sh --host 0.0.0.0 --port 8767
#   bash scripts/launch/run_any6d.sh --mesh /path/to/object.obj
#
# 环境变量:
#   ANY6D_CONDA_ENV   默认 Any6D（其次 foundationpose）
#   ANY6D_PORT        默认 8767
#   ANY6D_HOST        默认 127.0.0.1
#   ANY6D_ROOT        Any6D 源码根
#   ANY6D_PYTHON      覆盖 conda python
#   ANY6D_MESH        默认 mesh（可空）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER="${SCRIPT_DIR}/compat/any6d_worker.py"

ANY6D_CONDA_ENV="${ANY6D_CONDA_ENV:-Any6D}"
ANY6D_PORT="${ANY6D_PORT:-8767}"
ANY6D_HOST="${ANY6D_HOST:-127.0.0.1}"
ANY6D_MESH="${ANY6D_MESH:-}"
if [[ -z "${ANY6D_ROOT:-}" ]]; then
  if [[ -f "${SCRIPT_DIR}/Any6D/estimater.py" ]]; then
    ANY6D_ROOT="${SCRIPT_DIR}/Any6D"
  elif [[ -f "${SCRIPT_DIR}/../Any6D/estimater.py" ]]; then
    ANY6D_ROOT="$(cd "${SCRIPT_DIR}/../Any6D" && pwd)"
  else
    ANY6D_ROOT="${HOME}/Any6D"
  fi
fi
export ANY6D_ROOT

DO_CHECK=0
EXTRA_ARGS=()

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1; shift ;;
    --host) ANY6D_HOST="$2"; shift 2 ;;
    --port) ANY6D_PORT="$2"; shift 2 ;;
    --mesh) ANY6D_MESH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

resolve_python() {
  if [[ -n "${ANY6D_PYTHON:-}" && -x "${ANY6D_PYTHON}" ]]; then
    echo "${ANY6D_PYTHON}"
    return
  fi
  local candidates=(
    "/opt/miniconda3/envs/any6d/bin/python"
    "/opt/miniconda3/envs/Any6D/bin/python"
    "/opt/miniconda3/envs/${ANY6D_CONDA_ENV}/bin/python"
    "${HOME}/miniconda3/envs/${ANY6D_CONDA_ENV}/bin/python"
    "${HOME}/anaconda3/envs/${ANY6D_CONDA_ENV}/bin/python"
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/${ANY6D_CONDA_ENV}/bin/python"
    "${HOME}/miniconda3/envs/foundationpose/bin/python"
    "/opt/miniconda3/envs/foundationpose/bin/python"
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/foundationpose/bin/python"
  )
  for p in "${candidates[@]}"; do
    if [[ -x "$p" ]]; then
      echo "$p"
      return
    fi
  done
  command -v python3
}

PY="$(resolve_python)"
echo "ANY6D_ROOT=${ANY6D_ROOT}"
echo "Python=${PY}"

# 权重优先: models/FoundationPose/weights → eai/FoundationPose/weights
# 软链到 Any6D/foundationpose/weights（官方代码硬编码从此读）
A6_W="${ANY6D_ROOT}/foundationpose/weights"
MODELS_W="${FOUNDATIONPOSE_WEIGHTS:-/share_data/projects/mahjong/share/personal/liyichao/models/FoundationPose/weights}"
FP_W="${SCRIPT_DIR}/FoundationPose/weights"
if [[ -d "${ANY6D_ROOT}/foundationpose" ]]; then
  if [[ ! -e "${A6_W}" ]]; then
    if [[ -d "${MODELS_W}" ]]; then
      ln -s "${MODELS_W}" "${A6_W}" && echo "linked weights: ${A6_W} -> ${MODELS_W}" || true
    elif [[ -d "${FP_W}" ]]; then
      ln -s "${FP_W}" "${A6_W}" && echo "linked weights: ${A6_W} -> ${FP_W}" || true
    fi
  elif [[ -L "${A6_W}" ]]; then
    echo "weights: ${A6_W} -> $(readlink -f "${A6_W}" 2>/dev/null || readlink "${A6_W}")"
  fi
fi

if [[ "${DO_CHECK}" -eq 1 ]]; then
  exec "${PY}" "${WORKER}" --check
fi

CMD=("${PY}" "${WORKER}" --serve --host "${ANY6D_HOST}" --port "${ANY6D_PORT}")
if [[ -n "${ANY6D_MESH}" ]]; then
  CMD+=(--mesh "${ANY6D_MESH}")
fi
CMD+=("${EXTRA_ARGS[@]}")
echo "→ ${CMD[*]}"
exec "${CMD[@]}"
