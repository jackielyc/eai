#!/usr/bin/env python3
"""Canonical 1:1 mapping: model / policy stack → one miniconda3 env.

Every entry resolves to exactly::

    {MINICONDA3}/envs/{env_name}

No RoboDojo_cache / home-fallback / cross-stack candidates here.
Override a single path with env var ``EAI_MODEL_CONDA_ENV_<KEY>`` (upper key)
or ``<ENV_VAR>`` when listed.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

_LIY = "/share_data/projects/mahjong/share/personal/liyichao"
MINICONDA3 = os.environ.get(
    "EAI_MINICONDA3",
    f"{_LIY}/miniconda3",
)
ENVS_ROOT = os.path.join(MINICONDA3, "envs")

# key → conda env name under ENVS_ROOT
# One stack / one runtime family → exactly one env name.
MODEL_CONDA_ENVS: Dict[str, str] = {
    # LLM / VLM deploy & train
    "qwen_deploy": "eai",
    "qwen25_vl": "Qwen2.5-VL",
    "qwen25_omni": "Qwen2.5-Omni",
    "qwen3_omni": "Qwen3-Omni",
    "eai_train": "eai-train",
    "bagel": "bagel",
    "vllm": "vllm",
    # Policy stacks
    "openpi": "openpi",
    "lingbot_va": "lingbot_va",
    "vpp2": "vpp2",
    "starvla": "starvla",
    "rise": "rise",
    "psi_policy": "psi-policy",
    "rynnvalue": "rynnvalue",
    "robometer": "robometer-policy",
    # Sim / eval clients
    "robodojo": "RoboDojo",
    "embodiedevo": "embodiedevo",
    "aspire": "aspire",
    "rlinf": "RLinf",
    "libero": "libero",
    "molmospaces": "molmospaces",
    "isaaclab_arena": "IsaacLab-Arena",
    "env_isaaclab": "env_isaaclab",
    # Perception / misc
    "sam3": "sam3",
    "foundationpose": "foundationpose",
    "humanego": "humanego",
}

# Optional override env vars for a key (checked before EAI_MODEL_CONDA_ENV_*).
MODEL_CONDA_ENV_OVERRIDES: Dict[str, str] = {
    "lingbot_va": "LINGBOT_VA_CONDA_ENV",
    "vpp2": "VPP2_CONDA_ENV",
    "openpi": "OPENPI_CONDA_ENV",
    "rise": "RISE_CONDA_ENV",
    "sam3": "SAM3_CONDA_ENV",
    "foundationpose": "FP_CONDA_ENV",
    "humanego": "HUMANECO_CONDA_ENV",
    "psi_policy": "PSI_POLICY_CONDA_ENV",
    "rynnvalue": "RYNNVALUE_CONDA_ENV",
    "embodiedevo": "EMBODIEDEVO_CONDA_ENV",
    "aspire": "ASPIRE_CONDA_ENV",
}


def env_name(model_key: str) -> str:
    key = (model_key or "").strip()
    if key not in MODEL_CONDA_ENVS:
        raise KeyError(f"unknown model conda key: {model_key!r}")
    return MODEL_CONDA_ENVS[key]


def env_dir(model_key: str) -> str:
    """Absolute path to the single conda env for ``model_key``."""
    key = (model_key or "").strip()
    ov = MODEL_CONDA_ENV_OVERRIDES.get(key, "")
    if ov:
        raw = (os.environ.get(ov) or "").strip()
        if raw:
            # allow either env name or absolute path
            if os.path.sep in raw or raw.startswith("~"):
                return os.path.abspath(os.path.expanduser(raw))
            return os.path.join(ENVS_ROOT, raw)
    generic = (os.environ.get(f"EAI_MODEL_CONDA_ENV_{key.upper()}") or "").strip()
    if generic:
        if os.path.sep in generic or generic.startswith("~"):
            return os.path.abspath(os.path.expanduser(generic))
        return os.path.join(ENVS_ROOT, generic)
    return os.path.join(ENVS_ROOT, env_name(key))


def env_python(model_key: str) -> str:
    return os.path.join(env_dir(model_key), "bin", "python")


def resolve_python(model_key: str, prefer: str = "") -> Optional[str]:
    for cand in ((prefer or "").strip(), env_python(model_key)):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def status_all() -> Tuple[Tuple[str, str, str, bool], ...]:
    """Return rows of (key, env_name, path, python_ok)."""
    rows = []
    for key, name in sorted(MODEL_CONDA_ENVS.items()):
        path = env_dir(key)
        py = os.path.join(path, "bin", "python")
        ok = os.path.isfile(py) and os.access(py, os.X_OK)
        rows.append((key, name, path, ok))
    return tuple(rows)


def main() -> int:
    missing = 0
    for key, name, path, ok in status_all():
        mark = "OK" if ok else "MISSING"
        if not ok:
            missing += 1
        print(f"{mark:7}  {key:18} → {name:22}  {path}")
    print(f"\nmissing={missing} / total={len(MODEL_CONDA_ENVS)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
