#!/usr/bin/env python3
"""Run LingBot-Depth refinement for the eai viewer tab."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

EAI_DIR = Path(__file__).resolve().parent
LINGBOT_DEPTH_ROOT = Path(
    os.environ.get(
        "LINGBOT_DEPTH_ROOT",
        "/share_data/projects/mahjong/share/personal/liyichao/lingbot-depth",
    )
).expanduser()


def _ensure_repo() -> None:
    root = str(LINGBOT_DEPTH_ROOT)
    if not (LINGBOT_DEPTH_ROOT / "mdm" / "model" / "v2.py").is_file():
        raise FileNotFoundError(f"lingbot-depth 仓库不存在: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)


def _emit_result(payload: dict) -> None:
    print("[RESULT] " + json.dumps(payload, ensure_ascii=False), flush=True)


def _find_rgb(example_dir: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = example_dir / f"rgb{ext}"
        if candidate.is_file():
            return candidate
    return None


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.example is not None and str(args.example).strip() != "":
        example_dir = LINGBOT_DEPTH_ROOT / "examples" / str(args.example).strip()
        if not example_dir.is_dir():
            raise FileNotFoundError(f"示例目录不存在: {example_dir}")
        rgb = _find_rgb(example_dir)
        depth = example_dir / "raw_depth.png"
        intrinsics = example_dir / "intrinsics.txt"
        if rgb is None:
            raise FileNotFoundError(f"示例缺少 rgb 图像: {example_dir}")
        if not depth.is_file():
            raise FileNotFoundError(f"示例缺少 raw_depth.png: {example_dir}")
        if not intrinsics.is_file():
            raise FileNotFoundError(f"示例缺少 intrinsics.txt: {example_dir}")
        return rgb, depth, intrinsics

    rgb = Path(os.path.abspath(os.path.expanduser(args.rgb or "")))
    depth = Path(os.path.abspath(os.path.expanduser(args.depth or "")))
    intrinsics = Path(os.path.abspath(os.path.expanduser(args.intrinsics or "")))
    missing = []
    if not rgb.is_file():
        missing.append(f"rgb={rgb}")
    if not depth.is_file():
        missing.append(f"depth={depth}")
    if not intrinsics.is_file():
        missing.append(f"intrinsics={intrinsics}")
    if missing:
        raise FileNotFoundError("输入文件不存在: " + ", ".join(missing))
    return rgb, depth, intrinsics


def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-Depth worker")
    parser.add_argument("--example", default="", help="examples/<id> under lingbot-depth")
    parser.add_argument("--rgb", default="", help="RGB image path")
    parser.add_argument("--depth", default="", help="raw depth PNG (16-bit mm)")
    parser.add_argument("--intrinsics", default="", help="3x3 intrinsics txt/json")
    parser.add_argument(
        "--model",
        default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5",
        help="HF model id or local model.pt",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    args = parser.parse_args()

    try:
        _ensure_repo()
        import cv2
        import torch
        import trimesh
        from mdm.model.v2 import MDMModel

        # Reuse helpers from official example.py
        example_path = LINGBOT_DEPTH_ROOT / "example.py"
        if not example_path.is_file():
            raise FileNotFoundError(f"缺少 example.py: {example_path}")
        import importlib.util

        spec = importlib.util.spec_from_file_location("lingbot_depth_example", example_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 example.py")
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
    except Exception as exc:
        _emit_result({"ok": False, "error": f"导入 lingbot-depth 失败: {exc}"})
        return 1

    cache_dir = (args.cache_dir or "").strip() or str(EAI_DIR / ".cache" / "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)

    try:
        rgb_path, depth_path, intrinsics_path = _resolve_inputs(args)
    except Exception as exc:
        _emit_result({"ok": False, "error": str(exc)})
        return 2

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(os.path.abspath(os.path.expanduser(args.out)))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[lingbot_depth] model={args.model} device={device} "
        f"rgb={rgb_path} depth={depth_path}",
        flush=True,
    )

    try:
        image_np, image_tensor = example.preprocess_input_image(str(rgb_path), device)
        depth_np = example.load_depth_map(str(depth_path), scale=float(args.depth_scale))
        depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)
        h, w = image_np.shape[:2]
        if depth_np.shape[:2] != (h, w):
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_NEAREST)
            depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)
            print(f"[lingbot_depth] resized depth -> {w}x{h}", flush=True)

        print(
            f"[lingbot_depth] image={w}x{h} "
            f"depth_range="
            f"{(depth_np[depth_np > 0].min() if np.any(depth_np > 0) else 0):.3f}"
            f"-{depth_np.max():.3f}m",
            flush=True,
        )

        intrinsics = example.load_intrinsics(str(intrinsics_path), w, h)
        intrinsics_tensor = torch.tensor(
            intrinsics, dtype=torch.float32, device=device
        ).unsqueeze(0)

        print(f"[lingbot_depth] loading model…", flush=True)
        t0 = time.time()
        model = MDMModel.from_pretrained(args.model).to(device)
        load_s = time.time() - t0
        print(f"[lingbot_depth] model loaded in {load_s:.2f}s", flush=True)

        print(f"[lingbot_depth] inference…", flush=True)
        t1 = time.time()
        with torch.no_grad():
            output = model.infer(
                image_tensor,
                depth_in=depth_tensor,
                apply_mask=not args.no_mask,
                intrinsics=intrinsics_tensor,
            )
        infer_s = time.time() - t1

        depth_pred = output["depth"].squeeze().cpu().numpy()
        points_pred = output["points"].squeeze().cpu().numpy()
        print(
            f"[lingbot_depth] inference {infer_s:.3f}s "
            f"refined="
            f"{(depth_pred[depth_pred > 0].min() if np.any(depth_pred > 0) else 0):.3f}"
            f"-{depth_pred.max():.3f}m",
            flush=True,
        )

        np.save(out_dir / "depth_input.npy", depth_np)
        np.save(out_dir / "depth_refined.npy", depth_pred)

        depth_raw_color = example.depth_to_color_opencv(depth_np)
        depth_pred_color = example.depth_to_color_opencv(depth_pred)
        depth_concat = np.concatenate([depth_raw_color, depth_pred_color], axis=1)

        rgb_out = out_dir / "rgb.png"
        depth_in_out = out_dir / "depth_input.png"
        depth_ref_out = out_dir / "depth_refined.png"
        cmp_out = out_dir / "depth_comparison.png"
        ply_out = out_dir / "point_cloud.ply"

        cv2.imwrite(str(rgb_out), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(depth_in_out), depth_raw_color)
        cv2.imwrite(str(depth_ref_out), depth_pred_color)
        cv2.imwrite(str(cmp_out), depth_concat)

        valid_mask = np.isfinite(points_pred).all(axis=-1) & (points_pred[..., 2] > 0)
        verts = points_pred[valid_mask][::2]
        verts_color = image_np[valid_mask][::2]
        point_cloud = trimesh.PointCloud(verts, verts_color)
        point_cloud.export(ply_out)

        _emit_result(
            {
                "ok": True,
                "model": args.model,
                "device": str(device),
                "load_s": round(load_s, 3),
                "infer_s": round(infer_s, 3),
                "width": int(w),
                "height": int(h),
                "points": int(len(verts)),
                "rgb": str(rgb_out),
                "depth_input": str(depth_in_out),
                "depth_refined": str(depth_ref_out),
                "comparison": str(cmp_out),
                "point_cloud": str(ply_out),
                "out_dir": str(out_dir),
            }
        )
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        _emit_result({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
