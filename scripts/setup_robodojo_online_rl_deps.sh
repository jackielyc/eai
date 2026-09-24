#!/usr/bin/env bash
# 为 RLinf conda 环境补齐 RoboDojo/RISE online RL 所需的 openpi_value 依赖。
# 用法:
#   bash scripts/setup_robodojo_online_rl_deps.sh
#   bash scripts/setup_robodojo_online_rl_deps.sh /path/to/python
set -euo pipefail

PY="${1:-${RISE_PYTHON:-${RLINF_PYTHON:-/home/psibot/miniconda3/envs/RLinf/bin/python}}}"
if [[ ! -x "${PY}" ]]; then
  echo "错误: Python 不可执行: ${PY}" >&2
  exit 1
fi
PIP="$(dirname "${PY}")/pip"
if [[ ! -x "${PIP}" ]]; then
  echo "错误: 未找到 pip: ${PIP}" >&2
  exit 1
fi

echo "[setup] python=${PY}"
echo "[setup] pip=${PIP}"

# 避免 fsspec[gcs] / 完整 lerobot 依赖链（会强行升级 torch）
"${PIP}" install --no-cache-dir \
  'tqdm-loggable>=0.2' \
  'etils[epath]>=1.7' \
  'filelock>=3.16' \
  'flax==0.10.2' \
  'beartype==0.19.0' \
  'tyro>=0.9.5' \
  'dm-tree>=0.1.8' \
  'einops>=0.8.0' \
  'numpydantic>=1.6.6' \
  'augmax>=0.3.4' \
  'equinox>=0.11.8' \
  'jaxtyping==0.2.36' \
  'ml_collections==1.0.0' \
  'openpi-client' \
  'sentencepiece>=0.2.0' \
  'treescope>=0.1.7' \
  'flatbuffers>=24.3.25' \
  'imageio>=2.36.1' \
  'polars>=1.30.0' \
  'pytest' \
  'kornia' \
  'transformers==4.53.2' \
  'gcsfs>=2024.6.0'

# lerobot 非 online 热路径必需；若后续离线数据管线需要再装完整 HF lerobot。

RISE_ROOT="${RISE_ROOT:-/share_data/projects/mahjong/share/personal/liyichao/RoboDojo/XPolicyLab/policy/RISE/RISE}"
export PYTHONPATH="${RISE_ROOT}/policy_and_value/policy_online:${RISE_ROOT}/policy_and_value/policy_offline_and_value/src:${RISE_ROOT}/dynamics${PYTHONPATH:+:${PYTHONPATH}}"

# openpi 要求把 transformers_replace 覆盖进 site-packages/transformers（版本必须是 4.53.2）
TRANSFORMERS_DIR="$("${PY}" -c 'import os, transformers; print(os.path.dirname(transformers.__file__))')"
REPLACE_SRC="${RISE_ROOT}/policy_and_value/policy_offline_and_value/src/openpi_value/models_pytorch/transformers_replace"
echo "[setup] overlay transformers_replace -> ${TRANSFORMERS_DIR}"
cp -r "${REPLACE_SRC}/"* "${TRANSFORMERS_DIR}/"

"${PY}" - <<'PY'
mods = [
    "tqdm_loggable",
    "flax",
    "pytest",
    "openpi_value.shared.download",
    "openpi_value.transforms",
    "openpi_value.training.config",
    "openpi_value.training.checkpoints",
    "openpi_value.models_pytorch.pi0_pytorch",
    "rlinf.models.embodiment.openpi_action_model",
]
failed = []
for m in mods:
    try:
        __import__(m)
        print(f"OK  {m}")
    except Exception as exc:
        print(f"FAIL {m}: {exc}")
        failed.append(m)
import transformers
from transformers.models.siglip import check
ver = transformers.__version__
ok = check.check_whether_transformers_replace_is_installed_correctly()
print(f"{'OK' if ok else 'FAIL'} transformers_replace (version={ver})")
if not ok:
    failed.append("transformers_replace")
if failed:
    raise SystemExit(f"仍有 {len(failed)} 个模块导入失败")
print("[setup] openpi_value 依赖就绪")
PY
