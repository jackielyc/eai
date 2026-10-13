#!/usr/bin/env python3
"""Run one MolmoSpaces JSON benchmark with a passive viewer when possible.

Forwards argv to molmo_spaces/evaluation/eval_main.py after patching
launch_passive (same EGL fallback as the datagen wrapper) and forcing
use_passive_viewer on the eval config.

Default viewer camera can be front third-person or first-person. Override with:
  MOLMOSPACES_VIEWER_FIRST_PERSON=1     # head / exo first-person
  MOLMOSPACES_VIEWER_CAM=head_camera_rgb
  MOLMOSPACES_VIEWER_CAM=exo_camera_1
  MOLMOSPACES_VIEWER_CAM=front         # front third-person
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Defaults; Psi GUI sets FIRST_PERSON=1 + head_camera_rgb.
    os.environ.setdefault("MOLMOSPACES_VIEWER_FIRST_PERSON", "0")
    os.environ.setdefault("MOLMOSPACES_VIEWER_CAM", "front")
    # Default FORCE_EGL=1 for legacy TurboVNC safety; GUI may set 0 for
    # Psi-MolmoSpaces to prefer official MuJoCo GLFW viewer.
    os.environ.setdefault("MOLMOSPACES_FORCE_EGL", "1")
    # Keep last frame after run until window close / GUI stop.
    os.environ.setdefault("MOLMOSPACES_VIEWER_KEEP_OPEN", "1")
    os.environ.setdefault("EAI_DIR", str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("RYNNVALUE_HUD_BACKEND", "molmospaces")

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

    from molmospaces_viewer_cam import patch_pipeline_viewer_cam
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

    # MolmoSpaces only applies viewer_cam_dict when it has "camera"; free-cam
    # front third-person must be applied via this patch.
    patch_pipeline_viewer_cam()

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
                # First-person: prefer Psi head cam, then common aliases / exo.
                preferred = cam_name or "head_camera_rgb"
                cfg.viewer_cam_dict = {
                    "camera": preferred,
                    "camera_fallbacks": [
                        preferred,
                        "head_camera_rgb",
                        "head",
                        "head_rgb",
                        "exo_camera_1",
                    ],
                }
            else:
                # 正面第三人称（备用）：setup_viewer 里跟踪机身。
                cfg.viewer_cam_dict = {
                    "distance": float(os.environ.get("MOLMOSPACES_VIEWER_DISTANCE") or 4.0),
                    "azimuth": 90.0,
                    "elevation": float(
                        os.environ.get("MOLMOSPACES_VIEWER_ELEVATION") or -25.0
                    ),
                    "lookat": [0.0, 0.0, 0.7],
                    "mode": "front",
                }
        # 「只开界面」：保留选中任务的 robot/scene，换成 DummyPolicy（不抓取/不推理）。
        if os.environ.get("MOLMOSPACES_VIEWER_ONLY") == "1":
            from molmo_spaces.configs.policy_configs import DummyPolicyConfig

            cfg.policy_config = DummyPolicyConfig()
            cfg.end_on_success = False
            # Keep the window up until GUI stop / viewer close.
            try:
                horizon = int(getattr(cfg, "task_horizon", 0) or 0)
            except Exception:
                horizon = 0
            if horizon < 100000:
                cfg.task_horizon = 100000
            print(
                "[molmospaces-eval] viewer_only=1: DummyPolicy, keep robot/scene, "
                f"task_horizon={cfg.task_horizon}",
                flush=True,
            )
        return cfg

    eval_main.create_eval_config = _create_with_viewer
    sys.argv = [str(eval_py), *sys.argv[1:]]
    os.chdir(ms_root)
    eval_main.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
