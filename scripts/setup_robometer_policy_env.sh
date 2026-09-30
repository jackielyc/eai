#!/usr/bin/env bash
# 在独立 conda 环境 robometer-policy 中安装/更新 robometer-policy-learning 依赖。
# 用法:
#   bash scripts/setup_robometer_policy_env.sh
set -euo pipefail

EAI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/miniconda3}"
ENV_PREFIX="${ROBOMETER_CONDA_ENV:-${CONDA_ROOT}/envs/robometer-policy}"
ROOT="${ROBOMETER_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/robometer-policy-learning}"
CONDA_BIN="${CONDA_ROOT}/bin/conda"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo ">>> 创建 conda 环境: ${ENV_PREFIX}"
  export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${CONDA_ROOT}/pkgs_local}"
  mkdir -p "${CONDA_PKGS_DIRS}"
  "${CONDA_BIN}" create -y --solver=classic -p "${ENV_PREFIX}" python=3.11 pip
fi

PY="${ENV_PREFIX}/bin/python"
PIP="${ENV_PREFIX}/bin/pip"
export PATH="${ENV_PREFIX}/bin:${PATH}"

echo ">>> Python: ${PY} ($("${PY}" -V))"
"${PIP}" install -U pip uv

cd "${ROOT}"
# 子模块若为空，用 HTTPS 浅克隆
clone_one() {
  local name="$1" url="$2"
  if [[ -f "${name}/pyproject.toml" || -f "${name}/setup.py" ]]; then
    echo ">>> ${name} 已就绪"
    return 0
  fi
  echo ">>> 克隆 ${name}"
  rm -rf "${name}"
  git clone --filter=blob:none --depth 1 "${url}" "${name}"
}
clone_one LIBERO https://github.com/nakamotoo/LIBERO.git
clone_one dsrl_openpi https://github.com/jesbu1/dsrl_openpi.git
clone_one robometer https://github.com/robometer/robometer.git

echo ">>> uv sync（完整依赖，可能较久）"
export UV_PYTHON="${PY}"
unset UV_PROJECT_ENVIRONMENT || true
uv sync --python "${PY}" || {
  echo ">>> uv sync 失败，回退到核心 pip + editable 安装"
  "${PIP}" install --no-cache-dir \
    'torch' 'torchvision' \
    'hydra-core==1.3.2' 'omegaconf' \
    'gymnasium>=0.29' 'gym' \
    'h5py' 'numpy' 'pyyaml' \
    'wandb' 'tensorboard' 'loguru' 'tqdm' 'rich' \
    'pillow' 'opencv-python-headless' 'imageio' 'imageio-ffmpeg' \
    'transformers>=4.57.0,<5.0' 'sentence-transformers>=2.0.0,<4.0' 'huggingface-hub' \
    'grpcio-tools' 'easydict==1.9' 'cloudpickle==2.1.0' 'future==0.18.2' \
    'bddl==1.0.1' 'thop==0.1.1.post2209072238' 'termcolor' 'peft' 'pyrallis' \
    'codetiming>=1.4.0' 'uvicorn>=0.35.0' 'fastapi' 'einops' \
    'robosuite==1.4.0' 'mujoco' 'robomimic==0.2.0' 'metaworld'
  "${PIP}" install -e ./LIBERO
  "${PIP}" install -e ./robometer
  "${PIP}" install -e ./dsrl_openpi/packages/openpi-client
  "${PIP}" install -e . --no-deps
}

echo ">>> 导入自检"
PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" - <<'PY'
mods = [
    "robometer_policy_learning",
    "torch",
    "hydra",
    "openpi_client",
    "libero",
    "robometer",
]
for m in mods:
    try:
        __import__(m)
        print(f"OK  {m}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
PY

echo ">>> 完成。启动示例:"
echo "  bash ${EAI_DIR}/run_robometer_policy.sh --job train --config libero_online_rl --gui"
echo "  或在 GUI RoboMeter 页使用 Python=${ENV_PREFIX}/bin/python"
