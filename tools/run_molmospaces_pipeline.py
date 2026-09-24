#!/usr/bin/env python3
"""Run MolmoSpaces datagen pipeline with EGL passive viewer fallback.

TurboVNC / no-GLX: official mujoco.viewer.launch_passive aborts the process.
This wrapper patches launch_passive to an EGL+Tk Handle before importing the
pipeline, then forwards remaining CLI args to run_pipeline.py.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    here = Path(__file__).resolve().parent
    eai_root = here.parent
    # Prefer explicit MOLMOSPACES_ROOT, else sibling checkout.
    ms_root = Path(
        os.environ.get(
            "MOLMOSPACES_ROOT",
            str(eai_root.parent / "molmospaces"),
        )
    ).resolve()
    pipeline = ms_root / "scripts" / "datagen" / "run_pipeline.py"
    if not pipeline.is_file():
        print(f"[molmospaces-wrapper] pipeline not found: {pipeline}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(here))
    sys.path.insert(0, str(ms_root / "scripts" / "datagen"))

    from mujoco_egl_passive import patch_launch_passive

    backend = patch_launch_passive(force_egl=os.environ.get("MOLMOSPACES_FORCE_EGL") == "1")
    print(f"[molmospaces-wrapper] viewer_backend={backend}", flush=True)
    print(f"[molmospaces-wrapper] pipeline={pipeline}", flush=True)

    # Forward argv after this script name.
    sys.argv = [str(pipeline), *sys.argv[1:]]
    os.chdir(ms_root)
    runpy.run_path(str(pipeline), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
