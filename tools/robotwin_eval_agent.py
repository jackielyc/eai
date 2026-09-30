#!/usr/bin/env python3
"""RLinf RoboTwin eval entry with optional SAPIEN Viewer hold + RynnValue Live HUD.

Same as ``examples/embodiment/eval_embodied_agent.py``, but:

  - ``ROBOTWIN_HOLD_VIEWER=1``: keep SAPIEN Viewer until EAI「停止」
  - ``RYNNVALUE_FRAME_BUS=<dir>``: EnvWorker publishes RGB frames; driver runs
    Live HUD sidecar (OpenCV + reward_server) reading that bus
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)

# Ensure EAI tools/ is importable on driver + Ray workers (frame bus / bridge).
_TOOLS = pathlib.Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))


class HudEnvWorker(EnvWorker):
    """EnvWorker that publishes RoboTwin RGB frames to ``RYNNVALUE_FRAME_BUS``."""

    def init_worker(self):
        try:
            from rynnvalue_frame_bus import maybe_install_robotwin_hook_from_env

            maybe_install_robotwin_hook_from_env()
        except Exception as exc:  # noqa: BLE001
            print(f"[robotwin] RynnValue frame hook skipped: {exc}", flush=True)
        return super().init_worker()


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes", "YES")


def _maybe_start_hud_sidecar():
    bus = (os.environ.get("RYNNVALUE_FRAME_BUS") or "").strip()
    if not bus or not _truthy("RYNNVALUE_LIVE_HUD"):
        return None
    try:
        from rynnvalue_frame_bus import FrameBusHudSidecar
        from rynnvalue_sim_bridge import get_hud_profile
    except Exception as exc:  # noqa: BLE001
        print(f"[robotwin] Live HUD sidecar import failed: {exc}", flush=True)
        return None

    prof = get_hud_profile("robotwin")
    robot = (
        os.environ.get("RYNNVALUE_ROBOT_DESCRIPTION") or ""
    ).strip() or prof.robot_description
    camera = (
        os.environ.get("RYNNVALUE_CAMERA_DESCRIPTION") or ""
    ).strip() or prof.camera_description
    status = (os.environ.get("RYNNVALUE_STATUS_PATH") or "").strip() or str(
        pathlib.Path(bus) / "rynnvalue_live.json"
    )
    sidecar = FrameBusHudSidecar(
        bus,
        server_url=(
            os.environ.get("RYNNVALUE_SERVER_URL") or "http://127.0.0.1:8001"
        ).strip(),
        refresh_sec=float(os.environ.get("RYNNVALUE_REFRESH_SEC") or 1.0),
        num_frames=int(os.environ.get("RYNNVALUE_NUM_FRAMES") or 8),
        robot_description=robot,
        camera_description=camera,
        show_window=not _truthy("RYNNVALUE_NO_SHOW_WINDOW"),
        status_path=status,
        timeout_s=float(os.environ.get("RYNNVALUE_TIMEOUT_S") or 60.0),
    )
    sidecar.start()
    print(
        f"[robotwin] RynnValue Live HUD sidecar on bus={bus} "
        f"url={sidecar.hud.server_url}",
        flush=True,
    )
    return sidecar


@hydra.main(version_base="1.1", config_path=None, config_name=None)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    use_hud = bool((os.environ.get("RYNNVALUE_FRAME_BUS") or "").strip())
    env_cls = HudEnvWorker if use_hud else EnvWorker
    hud_sidecar = _maybe_start_hud_sidecar()

    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        component_placement = HybridComponentPlacement(cfg, cluster)

        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )
        env_placement = component_placement.get_strategy("env")
        env_group = env_cls.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        runner = EmbodiedEvalRunner(
            cfg=cfg,
            rollout=rollout_group,
            env=env_group,
        )

        runner.init_workers()
        runner.run()

        hold = _truthy("ROBOTWIN_HOLD_VIEWER")
        if not hold:
            return

        print(
            "[robotwin] 评测指标已写出；保持图形窗口，点击 EAI「停止」退出",
            flush=True,
        )
        try:
            env_group.hold_robotwin_viewer().wait()
        except BaseException as exc:  # noqa: BLE001 — Stop / Ray teardown
            print(f"[robotwin] hold ended ({type(exc).__name__}: {exc})", flush=True)
    finally:
        if hud_sidecar is not None:
            try:
                hud_sidecar.stop()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
