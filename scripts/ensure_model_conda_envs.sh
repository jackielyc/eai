#!/usr/bin/env bash
# Ensure every model stack in tools/model_conda_envs.py has exactly one
# miniconda3 env. Missing envs are created by cloning a related base env
# (or a minimal python create when no clone source fits).
set -euo pipefail

MINICONDA3="${EAI_MINICONDA3:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3}"
# shellcheck disable=SC1091
source "${MINICONDA3}/etc/profile.d/conda.sh"
conda activate base >/dev/null 2>&1 || true

ENVS="${MINICONDA3}/envs"
EAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ensure_clone() {
  local name="$1" base="$2"
  if [[ -x "${ENVS}/${name}/bin/python" ]]; then
    echo "[skip] ${name} already exists"
    return 0
  fi
  if [[ ! -x "${ENVS}/${base}/bin/python" ]]; then
    echo "[error] clone source missing: ${base}" >&2
    return 1
  fi
  echo "[create] ${name} ← clone ${base}"
  conda create -y -p "${ENVS}/${name}" --clone "${ENVS}/${base}"
}

ensure_py() {
  local name="$1" ver="$2"
  if [[ -x "${ENVS}/${name}/bin/python" ]]; then
    echo "[skip] ${name} already exists"
    return 0
  fi
  echo "[create] ${name} python=${ver}"
  conda create -y -p "${ENVS}/${name}" "python=${ver}"
}

echo "=== ensure model conda envs under ${ENVS} ==="

# Policy stacks that previously fell back onto RoboDojo / RLinf
ensure_clone lingbot_va RoboDojo
ensure_clone vpp2 RoboDojo
ensure_clone rise RLinf

# Standalone light envs (full deps via their install scripts when needed)
ensure_py humanego 3.11
ensure_py sam3 3.10
ensure_py foundationpose 3.10

echo
echo "=== status ==="
"${MINICONDA3}/bin/python" "${EAI_DIR}/tools/model_conda_envs.py" || true

echo
echo "Note: sam3 / foundationpose / humanego are minimal python envs."
echo "  Install deps when needed:"
echo "    bash ${EAI_DIR}/run_sam3.sh --install"
echo "    bash ${EAI_DIR}/run_foundationpose.sh --install"
echo "    (humanego) conda activate humanego && bash setup.sh in its repo"
