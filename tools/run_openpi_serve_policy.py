#!/usr/bin/env python3
"""Serve OpenPI with torch.compile disabled by default.

Pi0Config defaults to ``pytorch_compile_mode="max-autotune"``. On first
``sample_actions``, inductor autotune can block the server for tens of
seconds; the MolmoSpaces websocket client then hits keepalive timeout and
reconnects while the server is still compiling — so the arm never moves.

Set ``OPENPI_DISABLE_TORCH_COMPILE=0`` to keep stock compile behavior.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import runpy
import sys
from pathlib import Path


def _env_flag(name: str, default: str = "1") -> bool:
    return (os.environ.get(name, default) or default).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _disable_pytorch_compile(train_config):
    model = getattr(train_config, "model", None)
    if model is None or not hasattr(model, "pytorch_compile_mode"):
        return train_config
    if getattr(model, "pytorch_compile_mode", None) is None:
        return train_config
    return dataclasses.replace(
        train_config,
        model=dataclasses.replace(model, pytorch_compile_mode=None),
    )


def _patch_create_trained_policy() -> None:
    import openpi.policies.policy_config as policy_config

    if getattr(policy_config.create_trained_policy, "_eai_no_compile", False):
        return
    _orig = policy_config.create_trained_policy

    def _wrapped(train_config, checkpoint_dir, **kwargs):
        if _env_flag("OPENPI_DISABLE_TORCH_COMPILE", "1"):
            train_config = _disable_pytorch_compile(train_config)
            logging.info(
                "OpenPI serve: torch.compile disabled "
                "(OPENPI_DISABLE_TORCH_COMPILE=1; set 0 to restore max-autotune)"
            )
        return _orig(train_config, checkpoint_dir, **kwargs)

    _wrapped._eai_no_compile = True  # type: ignore[attr-defined]
    policy_config.create_trained_policy = _wrapped


def _resolve_serve_script() -> Path:
    override = (os.environ.get("OPENPI_SERVE_POLICY_SCRIPT") or "").strip()
    if override:
        return Path(override)
    root = (os.environ.get("OPENPI_ROOT") or "").strip()
    if root:
        cand = Path(root) / "scripts" / "serve_policy.py"
        if cand.is_file():
            return cand
    # Prefer sibling of this repo's default openpi checkout.
    here = Path(__file__).resolve()
    default = Path(
        "/share_data/projects/mahjong/share/personal/liyichao/psi-lab/openpi"
        "/scripts/serve_policy.py"
    )
    if default.is_file():
        return default
    # Last resort: walk PYTHONPATH for openpi package → repo root.
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if not entry:
            continue
        src = Path(entry)
        cand = src.parent / "scripts" / "serve_policy.py"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "serve_policy.py not found; set OPENPI_ROOT or OPENPI_SERVE_POLICY_SCRIPT"
    )


def _ensure_openpi_on_path(script: Path) -> None:
    """``scripts/serve_policy.py`` lives next to ``src/``; put ``src`` on sys.path."""
    root = script.resolve().parent.parent
    src = root / "src"
    if src.is_dir():
        src_s = str(src)
        if src_s not in sys.path:
            sys.path.insert(0, src_s)
        # Child env usually sets this; keep consistent for nested imports.
        existing = os.environ.get("PYTHONPATH") or ""
        parts = [p for p in existing.split(os.pathsep) if p]
        if src_s not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([src_s, *parts]) if parts else src_s
    os.environ.setdefault("OPENPI_ROOT", str(root))


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    os.environ.setdefault("OPENPI_DISABLE_TORCH_COMPILE", "1")
    script = _resolve_serve_script()
    _ensure_openpi_on_path(script)
    _patch_create_trained_policy()
    # Preserve tyro argv shape expected by serve_policy.
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
