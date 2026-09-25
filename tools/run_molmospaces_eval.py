#!/usr/bin/env python3
"""Run one MolmoSpaces JSON benchmark with a passive viewer when possible.

Forwards argv to molmo_spaces/evaluation/eval_main.py after patching
launch_passive (same EGL fallback as the datagen wrapper) and forcing
use_passive_viewer on the eval config.

Default viewer camera is front third-person (正对机器人). Override with:
  MOLMOSPACES_VIEWER_FIRST_PERSON=1     # head / exo first-person
  MOLMOSPACES_VIEWER_CAM=exo_camera_1
  MOLMOSPACES_VIEWER_CAM=wrist
  MOLMOSPACES_VIEWER_CAM=front         # front third-person (default)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Default: front third-person camera in the passive viewer.
    os.environ.setdefault("MOLMOSPACES_VIEWER_FIRST_PERSON", "0")
    os.environ.setdefault("MOLMOSPACES_VIEWER_CAM", "front")
    os.environ.setdefault("MOLMOSPACES_FORCE_EGL", "1")
    os.environ.setdefault("EAI_DIR", str(Path(__file__).resolve().parent.parent))

    here = Path(__file__).resolve().parent
    eai_root = here.parent
    ms_root = Path(
        os.environ.get(
            "MOLMOSPACES_ROOT",
            str(eai_root.parent / "molmospaces"),
        )
    ).resolve()
    eval_py = ms_root / "molmo_spaces" / "evaluation" / "eval_main.py"
    if not eval_py.is_file():
        print(f"[molmospaces-eval] eval_main not found: {eval_py}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(here))
    sys.path.insert(0, str(ms_root))

    from mujoco_egl_passive import patch_launch_passive

    backend = patch_launch_passive(force_egl=os.environ.get("MOLMOSPACES_FORCE_EGL") == "1")
    print(f"[molmospaces-eval] viewer_backend={backend}", flush=True)
    print(f"[molmospaces-eval] eval_main={eval_py}", flush=True)
    fp = os.environ.get("MOLMOSPACES_VIEWER_FIRST_PERSON", "0")
    cam = os.environ.get("MOLMOSPACES_VIEWER_CAM") or (
        "first_person(auto)"
        if fp not in ("0", "false", "False", "no")
        else "front_third_person"
    )
    print(f"[molmospaces-eval] viewer_cam={cam}", flush=True)

    import molmo_spaces.evaluation.eval_main as eval_main

    orig_create = eval_main.create_eval_config

    def _create_with_viewer(*args, **kwargs):
        cfg = orig_create(*args, **kwargs)
        if os.environ.get("MOLMOSPACES_EVAL_VIEWER", "1") == "1":
            cfg.use_passive_viewer = True
            force_fp = os.environ.get("MOLMOSPACES_VIEWER_FIRST_PERSON", "0") not in (
                "0",
                "false",
                "False",
                "no",
            )
            cam_name = (os.environ.get("MOLMOSPACES_VIEWER_CAM") or "").strip()
            if force_fp or (
                cam_name
                and cam_name.lower()
                not in ("front", "front_tp", "third_person", "third", "tp", "free", "3rd")
            ):
                cfg.viewer_cam_dict = {"camera": cam_name or "head"}
            else:
                cfg.viewer_cam_dict = {
                    "distance": 3.5,
                    "azimuth": 90.0,
                    "elevation": -20.0,
                    "lookat": [0.0, 0.0, 0.7],
                    "mode": "front",
                }
        return cfg

    eval_main.create_eval_config = _create_with_viewer
    sys.argv = [str(eval_py), *sys.argv[1:]]
    os.chdir(ms_root)
    eval_main.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
