#!/usr/bin/env python3
"""Run LingBot-World-V2 generate.py for the eai viewer tab."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EAI_DIR = Path(__file__).resolve().parent
LINGBOT_WORLD_ROOT = Path(
    os.environ.get(
        "LINGBOT_WORLD_ROOT",
        "/share_data/projects/mahjong/share/personal/liyichao/lingbot-world-v2",
    )
).expanduser()
DEFAULT_CKPT = "lingbot-world-v2-1.3b-causal-fast"
DEFAULT_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, "
    "surrounded by distant snow-capped mountains under a bright blue sky "
    "with drifting white clouds."
)


def _emit_result(payload: dict) -> None:
    print("[RESULT] " + json.dumps(payload, ensure_ascii=False), flush=True)


def _resolve_ckpt(ckpt_dir: str) -> Path:
    raw = (ckpt_dir or "").strip() or DEFAULT_CKPT
    path = Path(os.path.expanduser(raw))
    if not path.is_absolute():
        path = LINGBOT_WORLD_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"ckpt 目录不存在: {path}")
    # lightweight completeness check
    if not (path / "Wan2.1_VAE.pth").is_file() and not (
        path / "transformers"
    ).is_dir():
        raise FileNotFoundError(f"ckpt 看起来不完整: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-World-V2 worker")
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    parser.add_argument("--image", required=True)
    parser.add_argument("--action_path", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", required=True, help="output mp4 path")
    parser.add_argument("--size", default="480*832")
    parser.add_argument("--frame_num", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_mode", default="causal_fast")
    parser.add_argument("--local_attn_size", type=int, default=18)
    parser.add_argument("--sink_size", type=int, default=6)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    parser.add_argument("--no_t5_cpu", action="store_true")
    parser.add_argument("--offload_model", action="store_true", default=True)
    parser.add_argument("--no_offload_model", action="store_true")
    args = parser.parse_args()

    gen_script = LINGBOT_WORLD_ROOT / "generate.py"
    if not gen_script.is_file():
        _emit_result({"ok": False, "error": f"仓库不存在: {LINGBOT_WORLD_ROOT}"})
        return 1

    try:
        ckpt = _resolve_ckpt(args.ckpt_dir)
    except Exception as exc:
        _emit_result({"ok": False, "error": str(exc)})
        return 2

    image = Path(os.path.abspath(os.path.expanduser(args.image)))
    action = Path(os.path.abspath(os.path.expanduser(args.action_path)))
    if not image.is_file():
        _emit_result({"ok": False, "error": f"图像不存在: {image}"})
        return 2
    if not action.is_dir():
        _emit_result({"ok": False, "error": f"action_path 不存在: {action}"})
        return 2
    for need in ("poses.npy", "intrinsics.npy"):
        if not (action / need).is_file():
            _emit_result({"ok": False, "error": f"action_path 缺少 {need}: {action}"})
            return 2

    out_path = Path(os.path.abspath(os.path.expanduser(args.output)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_num = int(args.frame_num)
    if frame_num > 1 and (frame_num - 1) % 4 != 0:
        frame_num = max(1, ((frame_num - 1) // 4) * 4 + 1)

    t5_cpu = bool(args.t5_cpu) and not bool(args.no_t5_cpu)
    offload = bool(args.offload_model) and not bool(args.no_offload_model)

    cmd = [
        sys.executable,
        "-u",
        str(gen_script),
        "--task",
        "i2v-A14B",
        "--infer_mode",
        args.infer_mode,
        "--size",
        args.size,
        "--ckpt_dir",
        str(ckpt),
        "--image",
        str(image),
        "--action_path",
        str(action),
        "--prompt",
        args.prompt,
        "--frame_num",
        str(frame_num),
        "--base_seed",
        str(int(args.seed)),
        "--local_attn_size",
        str(int(args.local_attn_size)),
        "--sink_size",
        str(int(args.sink_size)),
        "--chunk_size",
        str(int(args.chunk_size)),
        "--ulysses_size",
        "1",
        "--save_file",
        str(out_path),
        "--save_dir",
        str(out_path.parent),
        "--offload_model",
        "true" if offload else "false",
    ]
    if t5_cpu:
        cmd.append("--t5_cpu")

    print(f"[lingbot_world] cwd={LINGBOT_WORLD_ROOT}", flush=True)
    print(f"[lingbot_world] ckpt={ckpt}", flush=True)
    print(f"[lingbot_world] image={image} action={action}", flush=True)
    print(f"[lingbot_world] frames={frame_num} size={args.size}", flush=True)

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = (
        f"{LINGBOT_WORLD_ROOT}"
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    # Prefer single GPU unless user set CUDA_VISIBLE_DEVICES
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("LINGBOT_WORLD_GPU", "0"))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(LINGBOT_WORLD_ROOT),
            env=env,
            check=False,
        )
    except Exception as exc:
        _emit_result({"ok": False, "error": str(exc)})
        return 1

    if proc.returncode != 0:
        _emit_result({"ok": False, "error": f"generate.py 退出码 {proc.returncode}"})
        return proc.returncode

    if not out_path.is_file():
        # fallback: newest mp4 in save_dir
        candidates = sorted(
            out_path.parent.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            out_path = candidates[0]
        else:
            _emit_result({"ok": False, "error": f"未生成视频: {args.output}"})
            return 1

    _emit_result(
        {
            "ok": True,
            "output": str(out_path),
            "ckpt_dir": str(ckpt),
            "image": str(image),
            "action_path": str(action),
            "frame_num": frame_num,
            "size": args.size,
            "infer_mode": args.infer_mode,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
