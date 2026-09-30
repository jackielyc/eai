#!/usr/bin/env python3
"""RLinf RoboTwin eval entry with optional SAPIEN Viewer hold-after-eval.

Same as ``examples/embodiment/eval_embodied_agent.py``, but when
``ROBOTWIN_HOLD_VIEWER=1`` (set by EAI GUI「图形窗口」), keeps the viewer
responsive after metrics are logged until SIGTERM/SIGINT (EAI「停止」).
"""
from __future__ import annotations

import json
import os

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


@hydra.main(version_base="1.1", config_path=None, config_name=None)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()

    hold = os.environ.get("ROBOTWIN_HOLD_VIEWER", "").strip() in ("1", "true", "True")
    if not hold:
        return

    print(
        "[robotwin] 评测指标已写出；保持图形窗口，点击 EAI「停止」退出",
        flush=True,
    )
    # Env worker pumps SAPIEN Viewer until process-group kill (EAI「停止」).
    # Driver must wait so the Ray cluster (and Viewer) stay alive.
    try:
        env_group.hold_robotwin_viewer().wait()
    except BaseException as exc:  # noqa: BLE001 — Stop / Ray teardown
        print(f"[robotwin] hold ended ({type(exc).__name__}: {exc})", flush=True)


if __name__ == "__main__":
    main()
