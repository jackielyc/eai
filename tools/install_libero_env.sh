#!/usr/bin/env bash
# Install a dedicated LIBERO client env for EAI 仿真评测.
# Usage: bash tools/install_libero_env.sh
set -euo pipefail

EAI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${LIBERO_ENV:-$EAI_DIR/.cache/envs/libero}"
OPENPI_ROOT="${OPENPI_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/psi-lab/openpi}"
LIBERO_SRC="${LIBERO_PATH:-$OPENPI_ROOT/third_party/libero}"
OPENPI_CLIENT="$OPENPI_ROOT/packages/openpi-client"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$EAI_DIR/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR" "$(dirname "$ENV")" "$EAI_DIR/.cache/libero"

# Never inherit GUI/ros PYTHONPATH into this env.
unset PYTHONPATH PYTHONHOME AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION || true
export PIP_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK=1
export PATH="/usr/bin:/bin:${PATH:-}"

BASEPY="${LIBERO_BASE_PYTHON:-}"
if [[ -z "$BASEPY" ]]; then
  for c in \
    /share_data/projects/mahjong/share/personal/liyichao/miniconda3/envs/openpi/bin/python \
    /share_data/projects/mahjong/share/personal/liyichao/miniconda3/bin/python \
    python3; do
    if [[ -x "$c" ]]; then
      BASEPY=$c
      break
    fi
  done
fi
if [[ -z "${BASEPY:-}" ]]; then
  echo "No base python found" >&2
  exit 1
fi
echo "[libero-env] base=$BASEPY"
echo "[libero-env] env=$ENV"

# Recreate if FORCE=1 or missing/broken.
if [[ "${FORCE:-0}" == "1" ]]; then
  rm -rf "$ENV"
fi
if [[ ! -x "$ENV/bin/python" ]]; then
  echo "[libero-env] creating venv…"
  "$BASEPY" -m venv --clear "$ENV"
fi
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"

echo "[libero-env] bootstrap pip…"
"$PY" -m ensurepip --upgrade 2>/dev/null || true
"$PY" -m pip install -U 'pip<26' setuptools wheel

# Pin stack to avoid pip backtracking forever on opencv.
# robosuite 1.4.1 needs mujoco>=2.3, numba, opencv, etc.
echo "[libero-env] install numpy / scientific…"
"$PIP" install --no-cache-dir 'numpy>=1.23,<2' 'scipy>=1.10,<1.14'

echo "[libero-env] install opencv (pinned)…"
"$PIP" install --no-cache-dir 'opencv-python==4.8.1.78'

echo "[libero-env] install mujoco / imageio…"
"$PIP" install --no-cache-dir 'mujoco>=2.3.0,<3.3' imageio imageio-ffmpeg tqdm pyyaml einops Pillow

echo "[libero-env] install robosuite==1.4.1…"
"$PIP" install --no-cache-dir 'robosuite==1.4.1'

echo "[libero-env] install torch (cpu, needed by libero.benchmark)…"
"$PIP" install --no-cache-dir 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cpu \
  || "$PIP" install --no-cache-dir 'torch==2.2.2'

echo "[libero-env] install libero helper deps…"
# Keep numpy<2 (opencv/robosuite binary wheels). Pin matplotlib stack accordingly.
"$PIP" install --no-cache-dir \
  'numpy>=1.23,<2' \
  easydict bddl cloudpickle gym termcolor \
  'matplotlib>=3.7,<3.9' 'contourpy>=1.0.1,<1.3'

# Do NOT pip install -e libero: openpi's tree lacks top-level __init__.py and
# editable finder maps wrongly. Use PYTHONPATH=third_party/libero (official README).
if [[ -d "$OPENPI_CLIENT" ]]; then
  echo "[libero-env] install openpi-client…"
  "$PIP" install -e "$OPENPI_CLIENT"
fi

# Seed non-interactive LIBERO config.yaml
export LIBERO_CONFIG_PATH="$EAI_DIR/.cache/libero"
PKG_ROOT="$LIBERO_SRC/libero/libero"
if [[ ! -d "$PKG_ROOT" ]]; then
  PKG_ROOT="$LIBERO_SRC/libero"
fi
if [[ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]]; then
  cat >"$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $PKG_ROOT
bddl_files: $PKG_ROOT/bddl_files
init_states: $PKG_ROOT/init_files
datasets: $LIBERO_SRC/datasets
assets: $PKG_ROOT/assets
EOF
  echo "[libero-env] wrote $LIBERO_CONFIG_PATH/config.yaml"
fi

echo "[libero-env] verify imports…"
# Official layout: PYTHONPATH points at third_party/libero (parent of package dir).
export PYTHONPATH="$LIBERO_SRC"
"$PY" - <<'PY'
import robosuite
import matplotlib
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv  # needs matplotlib
print("robosuite", robosuite.__version__)
print("matplotlib", matplotlib.__version__)
print("suites", sorted(benchmark.get_benchmark_dict().keys()))
print("OffScreenRenderEnv", OffScreenRenderEnv)
try:
    from openpi_client import websocket_client_policy  # noqa: F401
    print("openpi_client OK")
except Exception as e:
    print("openpi_client missing:", type(e).__name__, e)
PY

echo
echo "Done."
echo "  export LIBERO_PYTHON=$ENV/bin/python"
echo "  export LIBERO_CONFIG_PATH=$EAI_DIR/.cache/libero"
echo "  export PYTHONPATH=$LIBERO_SRC:\${PYTHONPATH:-}"
