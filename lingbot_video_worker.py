#!/usr/bin/env python3
"""Run LingBot-Video DiT inference for the eai viewer tab."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EAI_DIR = Path(__file__).resolve().parent
LINGBOT_VIDEO_ROOT = Path(
    os.environ.get(
        "LINGBOT_VIDEO_ROOT",
        "/share_data/projects/mahjong/share/personal/liyichao/lingbot-video",
    )
).expanduser()
DEFAULT_HF_REPO = "robbyant/lingbot-video-dense-1.3b"


def _emit_result(payload: dict) -> None:
    print("[RESULT] " + json.dumps(payload, ensure_ascii=False), flush=True)


def _ensure_repo() -> Path:
    infer = LINGBOT_VIDEO_ROOT / "scripts" / "inference.py"
    if not infer.is_file():
        raise FileNotFoundError(f"lingbot-video 仓库不存在: {LINGBOT_VIDEO_ROOT}")
    root = str(LINGBOT_VIDEO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return infer


def _ensure_model_dir(model_dir: str, cache_dir: Path) -> Path:
    model_dir = (model_dir or "").strip()
    if model_dir:
        path = Path(os.path.abspath(os.path.expanduser(model_dir)))
        if (path / "transformer").is_dir() and (
            (path / "model_index.json").is_file() or (path / "vae").is_dir()
        ):
            return path
        raise FileNotFoundError(f"模型目录无效（需含 transformer/）: {path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "lingbot-video-dense-1.3b"
    marker = target / "model_index.json"
    transformer = target / "transformer"
    # Incomplete prior downloads may leave model_index.json without weights.
    if marker.is_file() and transformer.is_dir():
        has_weights = any(transformer.glob("*.safetensors")) or any(
            transformer.glob("*.bin")
        )
        if has_weights:
            print(f"[lingbot_video] using cached model: {target}", flush=True)
            return target
        print(
            f"[lingbot_video] cached transformer 无权重，重新下载: {target}",
            flush=True,
        )
    elif marker.is_file() and not transformer.is_dir():
        print(
            f"[lingbot_video] 缓存不完整（缺 transformer/），重新下载: {target}",
            flush=True,
        )

    print(
        f"[lingbot_video] downloading {DEFAULT_HF_REPO} → {target} "
        f"（首次较大，请耐心等待）",
        flush=True,
    )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DEFAULT_HF_REPO,
        local_dir=str(target),
    )
    if not transformer.is_dir():
        raise FileNotFoundError(f"下载完成但缺少 transformer/: {target}")
    print(f"[lingbot_video] model ready: {target}", flush=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-Video worker")
    parser.add_argument("--model_dir", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--mode", required=True, choices=("t2i", "t2v", "ti2v"))
    parser.add_argument("--prompt_json", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_refiner", action="store_true")
    args = parser.parse_args()

    try:
        infer_script = _ensure_repo()
    except Exception as exc:
        _emit_result({"ok": False, "error": str(exc)})
        return 1

    hf_home = EAI_DIR / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home))
    os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_flash")

    cache_dir = Path(
        (args.cache_dir or "").strip()
        or str(EAI_DIR / ".cache" / "lingbot_video" / "models")
    )
    try:
        model_dir = _ensure_model_dir(args.model_dir, cache_dir)
    except Exception as exc:
        _emit_result({"ok": False, "error": f"模型准备失败: {exc}"})
        return 2

    if not args.prompt_json and not args.prompt:
        _emit_result({"ok": False, "error": "需要 --prompt_json 或 --prompt"})
        return 2

    out_path = Path(os.path.abspath(os.path.expanduser(args.output)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(infer_script),
        "--backend",
        args.backend or "diffusers",
        "--model_dir",
        str(model_dir),
        "--mode",
        args.mode,
        "--output",
        str(out_path),
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num_frames",
        str(max(1, int(args.num_frames))),
        "--steps",
        str(max(1, int(args.steps))),
        "--guidance_scale",
        str(args.guidance_scale),
        "--shift",
        str(args.shift),
        "--seed",
        str(int(args.seed)),
        "--fps",
        str(max(1, int(args.fps))),
        "--transformer_dtype",
        "bf16",
        "--text_encoder_dtype",
        "bf16",
        "--vae_dtype",
        "fp32",
    ]
    if args.prompt_json:
        cmd.extend(
            [
                "--prompt_json",
                os.path.abspath(os.path.expanduser(args.prompt_json)),
            ]
        )
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if args.image:
        cmd.extend(
            ["--image", os.path.abspath(os.path.expanduser(args.image))]
        )
    if args.run_refiner:
        cmd.append("--run_refiner")
        refined = out_path.with_name(out_path.stem + "_refined" + out_path.suffix)
        cmd.extend(["--refiner_output", str(refined)])

    print(f"[lingbot_video] cwd={LINGBOT_VIDEO_ROOT}", flush=True)
    print(f"[lingbot_video] model_dir={model_dir}", flush=True)
    print(f"[lingbot_video] cmd={' '.join(cmd[2:])}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{LINGBOT_VIDEO_ROOT}:{LINGBOT_VIDEO_ROOT / 'rewriter'}"
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_flash")
    env.setdefault("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "sdpa")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(LINGBOT_VIDEO_ROOT),
            env=env,
            check=False,
        )
    except Exception as exc:
        _emit_result({"ok": False, "error": str(exc)})
        return 1

    if proc.returncode != 0:
        _emit_result(
            {
                "ok": False,
                "error": f"inference 退出码 {proc.returncode}",
                "output": str(out_path) if out_path.is_file() else "",
            }
        )
        return proc.returncode

    if not out_path.is_file():
        _emit_result({"ok": False, "error": f"未生成输出文件: {out_path}"})
        return 1

    payload = {
        "ok": True,
        "mode": args.mode,
        "output": str(out_path),
        "model_dir": str(model_dir),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "steps": int(args.steps),
    }
    refined = out_path.with_name(out_path.stem + "_refined" + out_path.suffix)
    if refined.is_file():
        payload["refined"] = str(refined)
    _emit_result(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
