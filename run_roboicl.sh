#!/usr/bin/env bash
# 启动 RoboICL（dry-run / capture-only / 完整 rollout）。
# 由 eai viewer「RoboICL」页调用；也可手动：
#   bash run_roboicl.sh --task deposit_coin --shots 0 --mode dry_run
#   bash run_roboicl.sh --task deposit_coin --shots 0 --mode capture_only --gui
#   bash run_roboicl.sh --task deposit_coin --shots 0 --mode run --gui
set -euo pipefail

DEFAULT_ROOT="/share_data/projects/mahjong/share/personal/liyichao/RoboICL"
ROOT="${ROBOICL_ROOT:-${DEFAULT_ROOT}}"
POLICY_PY="${ROBOICL_POLICY_PYTHON:-}"
SIM_PY="${ROBOICL_SIM_PYTHON:-}"
TASK="deposit_coin"
SHOTS="0"
SEED="0"
LAYOUT="0"
GPU="0"
MODE="run"
PROFILE=""
DATA_ROOT=""
RESULTS_ROOT=""
REFERENCE=""
MODEL=""
API_KEY=""
LINK_ASSETS=""
ASSETS_CACHE=""
EXTRA_ARGS=()
GUI=0
DEFAULT_LINK_ASSETS="/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/Assets"

usage() {
  cat <<'EOF'
用法:
  bash run_roboicl.sh [选项]

选项:
  --task NAME           RoboDojo 任务名（默认 deposit_coin）
  --shots N             0=zero-shot，1=one-shot（默认 0）
  --seed N              随机种子（默认 0）
  --layout N            layout id（默认 0）
  --gpu N               CUDA 设备号（默认 0）
  --mode NAME           dry_run | capture_only | run | prepare_assets（默认 run）
  --profile PATH        protocol json（默认按 shots 选 zero/one-shot）
  --root PATH           RoboICL 仓库根
  --policy-python PATH  策略进程 Python（ROBOICL_POLICY_PYTHON）
  --sim-python PATH     仿真进程 Python（ROBOICL_SIM_PYTHON）
  --data-root PATH      data 根（含 Assets/）
  --results-root PATH   结果目录
  --reference PATH      one-shot 参考 bundle 目录
  --model NAME          Astra 模型名覆盖
  --api-key KEY         写入 ASTRA_API_KEY（可选；也可用环境变量）
  --link-assets PATH    prepare_assets：软链本地 Assets（默认 sibling RoboDojo/Assets）
  --assets-cache PATH   prepare_assets：走官方 fetch_assets.py 的 Git/LFS cache
  --gui                 弹出 Isaac Sim 窗口（去掉 --headless）
  -h, --help            显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="${2:-}"; shift 2 ;;
    --shots) SHOTS="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --layout) LAYOUT="${2:-}"; shift 2 ;;
    --gpu) GPU="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --policy-python) POLICY_PY="${2:-}"; shift 2 ;;
    --sim-python) SIM_PY="${2:-}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:-}"; shift 2 ;;
    --reference) REFERENCE="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --api-key) API_KEY="${2:-}"; shift 2 ;;
    --link-assets) LINK_ASSETS="${2:-}"; shift 2 ;;
    --assets-cache) ASSETS_CACHE="${2:-}"; shift 2 ;;
    --gui) GUI=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

ROOT="$(cd "${ROOT}" && pwd)"
if [[ ! -d "${ROOT}/roboicl" ]] || [[ ! -f "${ROOT}/roboicl/run.py" ]]; then
  echo "错误: 无效仓库（缺少 roboicl/run.py）: ${ROOT}" >&2
  exit 1
fi

MODE_NORM="$(echo "${MODE}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
case "${MODE_NORM}" in
  dry_run|dryrun|preflight) MODE_NORM="dry_run" ;;
  capture_only|capture|cameras) MODE_NORM="capture_only" ;;
  run|rollout|eval|full) MODE_NORM="run" ;;
  prepare_assets|prep_assets|assets|fetch_assets) MODE_NORM="prepare_assets" ;;
  *)
    echo "错误: 未知 --mode: ${MODE}" >&2
    exit 1
    ;;
esac

if [[ -z "${POLICY_PY}" ]]; then
  for cand in \
    "${ROBOICL_POLICY_PYTHON:-}" \
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/env_isaaclab/bin/python" \
    "${ROOT}/.venv/bin/python" \
    "${ROOT}/venv/bin/python"
  do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
      POLICY_PY="${cand}"
      break
    fi
  done
fi
if [[ -z "${POLICY_PY}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    POLICY_PY="$(command -v python3)"
  else
    echo "错误: 找不到 policy Python。请设置 ROBOICL_POLICY_PYTHON 或 --policy-python" >&2
    exit 1
  fi
fi

if [[ -z "${SIM_PY}" ]]; then
  for cand in \
    "${ROBOICL_SIM_PYTHON:-}" \
    "/share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/env_isaaclab/bin/python" \
    "${ROOT}/.venv/bin/python" \
    "${ROOT}/venv/bin/python"
  do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
      SIM_PY="${cand}"
      break
    fi
  done
fi
if [[ -z "${SIM_PY}" ]]; then
  SIM_PY="${POLICY_PY}"
fi

if [[ -z "${PROFILE}" ]]; then
  if [[ "${SHOTS}" == "1" ]]; then
    PROFILE="${ROOT}/configs/protocols/one_shot_j12_b12.json"
  else
    PROFILE="${ROOT}/configs/protocols/zero_shot_b25.json"
  fi
fi

DATA_ROOT="${DATA_ROOT:-${ROBOICL_DATA_ROOT:-${ROOT}/data}}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROBOICL_RESULTS_ROOT:-${ROOT}/results}}"
mkdir -p "${RESULTS_ROOT}"

export ROBOICL_ROOT="${ROOT}"
export ROBOICL_CODE_ROOT="${ROOT}"
export ROBOICL_DATA_ROOT="${DATA_ROOT}"
export ROBOICL_RESULTS_ROOT="${RESULTS_ROOT}"
export ROBOICL_POLICY_PYTHON="${POLICY_PY}"
export ROBOICL_SIM_PYTHON="${SIM_PY}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset PYTHONHOME || true

# IsaacLab defaults logs to $TMPDIR/isaaclab/logs. Shared /tmp/isaaclab is often
# root-owned / contested and can raise PermissionError on FileHandler.
ROBOICL_TMPDIR="${ROBOICL_TMPDIR:-${RESULTS_ROOT}/.tmpdir}"
mkdir -p "${ROBOICL_TMPDIR}/isaaclab/logs"
export TMPDIR="${ROBOICL_TMPDIR}"
export TEMP="${ROBOICL_TMPDIR}"
export TMP="${ROBOICL_TMPDIR}"
echo "[roboicl] TMPDIR=${TMPDIR}"

if [[ -n "${API_KEY}" ]]; then
  export ASTRA_API_KEY="${API_KEY}"
fi

# Clash Verge Sidecar：本环境无 systemd，系统代理/TUN 不可靠；有 7897 则注入 HTTPS_PROXY。
_CLASH_PROXY_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/clash_proxy_env.sh"
if [[ -f "${_CLASH_PROXY_HELPER}" ]]; then
  # shellcheck disable=SC1090
  source "${_CLASH_PROXY_HELPER}" || true
fi
unset _CLASH_PROXY_HELPER
if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
  echo "[roboicl] proxy=${HTTPS_PROXY:-${https_proxy}}"
fi

POLICY_BIN_DIR="$(cd "$(dirname "${POLICY_PY}")" && pwd)"
SIM_BIN_DIR="$(cd "$(dirname "${SIM_PY}")" && pwd)"
export PATH="${POLICY_BIN_DIR}:${SIM_BIN_DIR}:${PATH}"

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

if [[ -f "${ROOT}/setup/paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/setup/paths.sh"
fi

echo "[roboicl] log_dir=${RESULTS_ROOT}"
echo "[roboicl] mode=${MODE_NORM} task=${TASK} shots=${SHOTS} gui=${GUI}"
echo "[roboicl] root=${ROOT}"
echo "[roboicl] policy_python=${POLICY_PY}"
echo "[roboicl] sim_python=${SIM_PY}"
echo "[roboicl] profile=${PROFILE}"
echo "[roboicl] data_root=${DATA_ROOT}"
echo "[roboicl] results_root=${RESULTS_ROOT}"

if [[ "${MODE_NORM}" == "prepare_assets" ]]; then
  mkdir -p "${DATA_ROOT}"
  if [[ -n "${ASSETS_CACHE}" ]]; then
    echo "[roboicl] fetching locked Assets via setup/fetch_assets.py …"
    echo "$ ${POLICY_PY} setup/fetch_assets.py --data-root ${DATA_ROOT} --cache ${ASSETS_CACHE}"
    cd "${ROOT}"
    exec "${POLICY_PY}" setup/fetch_assets.py --data-root "${DATA_ROOT}" --cache "${ASSETS_CACHE}"
  fi
  SRC="${LINK_ASSETS:-${DEFAULT_LINK_ASSETS}}"
  if [[ ! -d "${SRC}" ]]; then
    echo "错误: 本地 Assets 不存在: ${SRC}" >&2
    echo "请传 --link-assets /path/to/Assets，或 --assets-cache /path/to/cache 走官方下载。" >&2
    exit 1
  fi
  TARGET="${DATA_ROOT}/Assets"
  if [[ -e "${TARGET}" || -L "${TARGET}" ]]; then
    if [[ -L "${TARGET}" ]]; then
      echo "[roboicl] Assets 已存在软链: ${TARGET} -> $(readlink "${TARGET}")"
    else
      echo "错误: ${TARGET} 已存在且不是软链；请先清理后再 prepare_assets" >&2
      exit 1
    fi
  else
    ln -s "${SRC}" "${TARGET}"
    echo "[roboicl] Linked ${TARGET} -> ${SRC}"
  fi
  "${POLICY_PY}" - <<PY
from pathlib import Path
import json
data_root = Path("${DATA_ROOT}").resolve()
code = Path("${ROOT}").resolve()
generated = []
for template in sorted((data_root / "Assets/Robots").glob("*/curobo_tmp.yml")):
    output = template.with_name("curobo.yml")
    content = template.read_text(encoding="utf-8").replace("\${ASSETS_PATH}", str(data_root))
    if (not output.exists()) or output.read_text(encoding="utf-8") != content:
        output.write_text(content, encoding="utf-8")
    generated.append(str(output.relative_to(data_root)))
lock = json.loads((code / "configs/data.lock.json").read_text(encoding="utf-8"))["assets"]
manifest = {
    "schema": "roboicl.assets.install.v1",
    "repo_id": lock["repo_id"],
    "revision": lock["revision"],
    "scope": "complete-tree",
    "include": lock["include"],
    "assets_tree_sha256": None,
    "assets": str(data_root / "Assets"),
    "cache": str(Path("${SRC}").resolve().parent),
    "generated": generated,
    "note": "local bootstrap symlink; prefer setup/fetch_assets.py for locked HF revision",
}
(data_root / ".roboicl-assets.json").write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
print("[roboicl] wrote", data_root / ".roboicl-assets.json")
print(json.dumps(manifest, indent=2))
PY
  LAYOUT="${DATA_ROOT}/Assets/Eval_Layout/RoboDojo/arx_x5/0/deposit_coin_0.json"
  if [[ ! -f "${LAYOUT}" ]]; then
    echo "警告: 未找到示例 layout: ${LAYOUT}" >&2
  else
    echo "[roboicl] layout ok: ${LAYOUT}"
  fi
  exit 0
fi

if [[ ! -f "${DATA_ROOT}/Assets/Eval_Layout/RoboDojo/arx_x5/${SEED}/${TASK}_${LAYOUT}.json" ]]; then
  echo "错误: 缺少 layout JSON（Assets 未准备）:" >&2
  echo "  ${DATA_ROOT}/Assets/Eval_Layout/RoboDojo/arx_x5/${SEED}/${TASK}_${LAYOUT}.json" >&2
  echo "请先运行:" >&2
  echo "  bash run_roboicl.sh --mode prepare_assets" >&2
  echo "或官方拉取:" >&2
  echo "  bash run_roboicl.sh --mode prepare_assets --assets-cache /path/to/robodojo-data-cache" >&2
  exit 1
fi

CMD=(
  -u -m roboicl.run
  "${TASK}"
  --shots "${SHOTS}"
  --seed "${SEED}"
  --layout "${LAYOUT}"
  --gpu "${GPU}"
  --profile "${PROFILE}"
  --data-root "${DATA_ROOT}"
  --results-root "${RESULTS_ROOT}"
  --policy-python "${POLICY_PY}"
  --sim-python "${SIM_PY}"
)
if [[ -n "${REFERENCE}" ]]; then
  CMD+=(--reference "${REFERENCE}")
fi
if [[ -n "${MODEL}" ]]; then
  CMD+=(--model "${MODEL}")
fi
if [[ "${MODE_NORM}" == "dry_run" ]]; then
  CMD+=(--dry-run)
elif [[ "${MODE_NORM}" == "capture_only" ]]; then
  CMD+=(--capture-only)
fi
if [[ "${GUI}" == "1" ]]; then
  CMD+=(--gui)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "$ cd ${ROOT} && ${POLICY_PY} ${CMD[*]}"
cd "${ROOT}"
exec "${POLICY_PY}" "${CMD[@]}"
