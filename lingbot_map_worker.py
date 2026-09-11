#!/usr/bin/env python3
"""Ensure LingBot-Map checkpoint, then run official demo.py (viser viewer)."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

EAI_DIR = Path(__file__).resolve().parent
LINGBOT_MAP_ROOT = Path(
    os.environ.get(
        "LINGBOT_MAP_ROOT",
        "/share_data/projects/mahjong/share/personal/liyichao/lingbot-map",
    )
).expanduser()
DEFAULT_HF_REPO = "robbyant/lingbot-map"
DEFAULT_CKPT_NAME = "lingbot-map.pt"


def _ensure_repo() -> Path:
    demo = LINGBOT_MAP_ROOT / "demo.py"
    if not demo.is_file():
        raise FileNotFoundError(f"lingbot-map 仓库不存在: {LINGBOT_MAP_ROOT}")
    root = str(LINGBOT_MAP_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return demo


def _resolve_model_path(model: str, cache_dir: Path) -> str:
    model = (model or "").strip()
    if model and Path(model).expanduser().is_file():
        return str(Path(model).expanduser().resolve())

    # Treat bare filename / empty as HF download of default ckpt
    filename = DEFAULT_CKPT_NAME
    if model and not model.startswith("robbyant/") and model.endswith(".pt"):
        filename = os.path.basename(model)
    elif model.startswith("robbyant/"):
        # allow robbyant/lingbot-map:lingbot-map-long.pt
        if ":" in model:
            _, filename = model.split(":", 1)
        # else default filename in that repo
    elif model in ("lingbot-map.pt", "lingbot-map-long.pt", "lingbot-map-stage1.pt"):
        filename = model

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_fallback = cache_dir / filename
    if local_fallback.is_file():
        print(f"[lingbot_map] using cached checkpoint: {local_fallback}", flush=True)
        return str(local_fallback)

    print(
        f"[lingbot_map] downloading {DEFAULT_HF_REPO}/{filename} → {cache_dir}",
        flush=True,
    )
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DEFAULT_HF_REPO,
        filename=filename,
        local_dir=str(cache_dir),
    )
    print(f"[lingbot_map] checkpoint ready: {path}", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-Map demo launcher for eai")
    parser.add_argument("--model", default="", help="local .pt or HF filename")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--video_path", default="")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--first_k", type=int, default=0, help="0 = all frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mode", default="streaming", choices=("streaming", "windowed"))
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--no_use_sdpa", action="store_true")
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=64)
    args, unknown = parser.parse_known_args()

    try:
        demo_path = _ensure_repo()
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return 1

    cache_dir = Path(
        (args.cache_dir or "").strip()
        or str(EAI_DIR / ".cache" / "lingbot_map" / "models")
    )
    hf_home = EAI_DIR / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home))

    try:
        model_path = _resolve_model_path(args.model, cache_dir)
    except Exception as exc:
        print(f"[ERROR] 无法准备模型: {exc}", flush=True)
        return 2

    if not args.image_folder and not args.video_path:
        print("[ERROR] 需要 --image_folder 或 --video_path", flush=True)
        return 2

    use_sdpa = bool(args.use_sdpa) and not bool(args.no_use_sdpa)
    # Prefer SDPA unless flashinfer is available and user passed --no_use_sdpa
    if not use_sdpa:
        try:
            import flashinfer  # noqa: F401
        except Exception:
            print("[lingbot_map] flashinfer 不可用，回退 --use_sdpa", flush=True)
            use_sdpa = True

    demo_argv = [
        str(demo_path),
        "--model_path",
        model_path,
        "--port",
        str(int(args.port)),
        "--mode",
        args.mode,
        "--stride",
        str(max(1, int(args.stride))),
        "--fps",
        str(max(1, int(args.fps))),
        "--conf_threshold",
        str(args.conf_threshold),
        "--downsample_factor",
        str(max(1, int(args.downsample_factor))),
        "--window_size",
        str(max(1, int(args.window_size))),
    ]
    if args.image_folder:
        demo_argv.extend(
            ["--image_folder", os.path.abspath(os.path.expanduser(args.image_folder))]
        )
    if args.video_path:
        demo_argv.extend(
            ["--video_path", os.path.abspath(os.path.expanduser(args.video_path))]
        )
    if int(args.first_k) > 0:
        demo_argv.extend(["--first_k", str(int(args.first_k))])
    if args.mask_sky:
        demo_argv.append("--mask_sky")
    if use_sdpa:
        demo_argv.append("--use_sdpa")
    demo_argv.extend(unknown)

    print(
        f"[lingbot_map] starting demo port={args.port} "
        f"folder={args.image_folder or '-'} video={args.video_path or '-'}",
        flush=True,
    )
    print(f"[VISER] http://127.0.0.1:{int(args.port)}", flush=True)
    print(f"[lingbot_map] argv: {' '.join(demo_argv[1:])}", flush=True)

    # matplotlib>=3.9 removed cm.get_cmap; lingbot-map vis still calls it.
    try:
        import matplotlib
        import matplotlib.cm as cm

        if not hasattr(cm, "get_cmap"):
            cm.get_cmap = matplotlib.colormaps.__getitem__  # type: ignore[attr-defined]
    except Exception as exc:
        print(f"[lingbot_map] matplotlib compat patch skipped: {exc}", flush=True)

    # Run from repo root so relative skyseg.onnx / examples resolve
    os.chdir(str(LINGBOT_MAP_ROOT))
    sys.argv = demo_argv
    runpy.run_path(str(demo_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
