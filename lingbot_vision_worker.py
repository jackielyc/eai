#!/usr/bin/env python3
"""Run LingBot-Vision PCA visualization for the eai viewer tab."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EAI_DIR = Path(__file__).resolve().parent
LINGBOT_VISION_ROOT = Path(
    os.environ.get(
        "LINGBOT_VISION_ROOT",
        "/share_data/projects/mahjong/share/personal/liyichao/lingbot-vision",
    )
).expanduser()


def _ensure_repo() -> None:
    root = str(LINGBOT_VISION_ROOT)
    if not (LINGBOT_VISION_ROOT / "lingbot_vision" / "__init__.py").is_file():
        raise FileNotFoundError(f"lingbot-vision 仓库不存在: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)


def _pca_rgb(patch_tokens, h: int, w: int) -> np.ndarray:
    x = patch_tokens[0].detach().float().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    rgb = x @ vt[:3].T
    rgb = rgb.reshape(h, w, 3)
    lo = np.percentile(rgb, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 99, axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = Image.fromarray(img)
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.size[0], 28), fill=(0, 0, 0))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((7, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(out)


def _save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_rgb).save(str(path))


def _emit_result(payload: dict) -> None:
    print("[RESULT] " + json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-Vision PCA worker")
    parser.add_argument("--input", required=True, help="RGB image path")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--variant",
        default="small",
        choices=("small", "base", "large", "giant"),
    )
    parser.add_argument("--ckpt", default="", help="optional local .pt checkpoint")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mode", default="square", choices=("square", "shortest"))
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="")
    parser.add_argument("--cache-dir", default="")
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input))
    if not os.path.isfile(input_path):
        _emit_result({"ok": False, "error": f"输入图像不存在: {input_path}"})
        return 2

    try:
        _ensure_repo()
        import torch
        from lingbot_vision import (
            extract_patch_tokens,
            load_backbone,
            load_backbone_state,
            load_config,
            load_image,
            load_pretrained_backbone,
        )
        from lingbot_vision.loader import PRETRAINED_VARIANTS, _packaged_config_path
    except Exception as exc:
        _emit_result({"ok": False, "error": f"导入 lingbot-vision 失败: {exc}"})
        return 1

    device = (args.device or "").strip() or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cache_dir = (args.cache_dir or "").strip() or str(
        EAI_DIR / ".cache" / "huggingface"
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)

    print(
        f"[lingbot_vision] variant={args.variant} device={device} "
        f"input={input_path} size={args.size}",
        flush=True,
    )

    try:
        ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt)) if args.ckpt else ""
        if ckpt_path and os.path.isfile(ckpt_path):
            spec = PRETRAINED_VARIANTS[args.variant]
            cfg_path = _packaged_config_path(spec["config_file"])
            if cfg_path is None:
                raise FileNotFoundError(spec["config_file"])
            from lingbot_vision.loader import _resolve_dtype

            print(f"[lingbot_vision] local ckpt={ckpt_path}", flush=True)
            cfg = load_config(str(cfg_path))
            ckpt = load_backbone_state(ckpt_path)
            resolved_dtype = _resolve_dtype(args.dtype, device)
            backbone, embed_dim = load_backbone(
                cfg, ckpt, device=device, dtype=resolved_dtype
            )
            del ckpt
        else:
            print("[lingbot_vision] loading pretrained backbone (HF if needed)…", flush=True)
            backbone, embed_dim = load_pretrained_backbone(
                variant=args.variant,
                device=device,
                dtype=args.dtype,
                cache_dir=cache_dir,
            )

        img_norm, img_rgb, (H, W) = load_image(
            input_path,
            size=args.size,
            patch_size=backbone.patch_size,
            mode=args.mode,
        )
        dtype = next(backbone.parameters()).dtype
        patch_tokens, (h, w) = extract_patch_tokens(backbone, img_norm, device, dtype)
        pca = _pca_rgb(patch_tokens, h, w)
        pca_up = np.asarray(
            Image.fromarray(pca).resize((W, H), resample=Image.NEAREST)
        )
        panel = np.concatenate(
            [
                _label(img_rgb, f"input {H}x{W}"),
                _label(pca_up, f"patch PCA {h}x{w}"),
            ],
            axis=1,
        )

        out_dir = Path(os.path.abspath(os.path.expanduser(args.out)))
        stem = Path(input_path).stem
        pca_path = out_dir / f"{stem}_pca.png"
        panel_path = out_dir / f"{stem}_panel.png"
        input_out = out_dir / f"{stem}_input.png"
        _save_rgb(pca_path, pca_up)
        _save_rgb(panel_path, panel)
        _save_rgb(input_out, img_rgb)
        print(f"[lingbot_vision] wrote {panel_path}", flush=True)
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        _emit_result(
            {
                "ok": True,
                "input": str(input_out),
                "pca": str(pca_path),
                "panel": str(panel_path),
                "grid": [int(h), int(w)],
                "embed_dim": int(embed_dim),
                "variant": args.variant,
                "device": str(device),
            }
        )
        return 0
    except Exception as exc:
        _emit_result({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
