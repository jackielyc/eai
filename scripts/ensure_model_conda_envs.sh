#!/usr/bin/env bash
# Ensure every model stack in tools/model_conda_envs.py has exactly one
# real miniconda3 env (no soft/hard links between env dirs).
# Missing envs: full conda --clone --copy, or minimal python create.
set -euo pipefail

MINICONDA3="${EAI_MINICONDA3:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3}"
CONDA_BIN="${MINICONDA3}/bin/conda"
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "[error] conda not found: ${CONDA_BIN}" >&2
  exit 1
fi

ENVS="${MINICONDA3}/envs"
EAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Reject symlink env dirs; force a real independent install.
reject_symlink_env() {
  local name="$1" path="${ENVS}/${name}"
  if [[ -L "${path}" ]]; then
    echo "[fix] ${name} is a symlink -> $(readlink "${path}"); removing"
    rm -f "${path}"
  fi
}

env_ready() {
  local name="$1" path="${ENVS}/${name}"
  [[ ! -L "${path}" && -x "${path}/bin/python" ]]
}

ensure_clone() {
  local name="$1" base="$2"
  reject_symlink_env "${name}"
  if env_ready "${name}"; then
    echo "[skip] ${name} already exists (real env)"
    return 0
  fi
  if [[ -d "${ENVS}/${name}" ]]; then
    echo "[clean] removing incomplete ${name}"
    rm -rf "${ENVS}/${name}"
  fi
  if [[ ! -x "${ENVS}/${base}/bin/python" ]]; then
    echo "[error] clone source missing: ${base}" >&2
    return 1
  fi
  # --copy: do not hard-link packages from the source env
  echo "[create] ${name} ← clone --copy ${base}"
  "${CONDA_BIN}" create -y -p "${ENVS}/${name}" --clone "${ENVS}/${base}" --copy
}

ensure_py() {
  local name="$1" ver="$2"
  reject_symlink_env "${name}"
  if env_ready "${name}"; then
    echo "[skip] ${name} already exists (real env)"
    return 0
  fi
  if [[ -d "${ENVS}/${name}" ]]; then
    echo "[clean] removing incomplete ${name}"
    rm -rf "${ENVS}/${name}"
  fi
  echo "[create] ${name} python=${ver}"
  "${CONDA_BIN}" create -y -p "${ENVS}/${name}" "python=${ver}" --copy
}

echo "=== ensure model conda envs under ${ENVS} (real copy, no links) ==="

# Policy stacks that previously fell back onto RoboDojo / RLinf
ensure_clone lingbot_va RoboDojo
ensure_clone vpp2 RoboDojo
ensure_clone rise RLinf

# Standalone light envs (full deps via their install scripts when needed)
ensure_py humanego 3.11
ensure_py sam3 3.10
ensure_py foundationpose 3.10
ensure_py aspire 3.11

echo
echo "=== status ==="
"${MINICONDA3}/bin/python" "${EAI_DIR}/tools/model_conda_envs.py" || true

echo
echo "Note: sam3 / foundationpose / humanego / aspire are minimal python envs."
echo "  Install deps when needed:"
echo "    bash ${EAI_DIR}/run_sam3.sh --install"
echo "    bash ${EAI_DIR}/run_foundationpose.sh --install"
echo "    (humanego) conda activate humanego && bash setup.sh in its repo"
echo "    (aspire) viewer ASPIRE 页点「安装 host 依赖」，或 python -c 'from aspire.catalog import ensure_host_env; ensure_host_env()'"
