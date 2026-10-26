"""Isolated GraspGen adapter. Run in the official GraspGen environment.

Returns model-frame grasp candidates, never robot commands. The configured
gripper must match the checkpoint; this does not retarget to a dexterous hand.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def infer(points, config, gripper):
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    cfg = load_grasp_cfg(config)
    if str(cfg.data.gripper_name) != gripper:
        raise ValueError(f"手型不匹配: checkpoint={cfg.data.gripper_name}, requested={gripper}")
    sampler = GraspGenSampler(cfg)
    poses, scores = GraspGenSampler.run_inference(
        points, sampler, num_grasps=100, topk_num_grasps=20,
        min_grasps=1, max_tries=1,
    )
    poses = poses.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy().reshape(-1)
    order = np.argsort(scores)[::-1][:20]
    return {"ok": True, "model": "GraspGen", "gripper": gripper,
            "collision_checked": False, "ik_checked": False,
            "candidates": [{"T_camera_grasp": poses[i].tolist(),
                            "model_score": float(scores[i])} for i in order]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gripper", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        points = np.load(args.points, allow_pickle=False)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
            raise ValueError("需要至少 100 个有限的米制 XYZ 点")
        result = infer(points.astype(np.float32), args.config, args.gripper)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    Path(args.output).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
