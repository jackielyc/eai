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
from perception_geometry import sanitize_depth, valid_depth_mask, DEPTH_MIN_M, DEPTH_MAX_M
from lingbot_depth_io import align_rgb_to_depth, load_depth_input, prediction_to_cloud, validate_intrinsics

EAI_DIR = Path(__file__).resolve().parent
LINGBOT_DEPTH_ROOT = Path(
    os.environ.get(
        "LINGBOT_DEPTH_ROOT",
        str(EAI_DIR.parent / "lingbot-depth"),
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


def _resolve_model_checkpoint(model: str) -> str:
    """Prefer a prepared local checkpoint for the viewer's official model IDs."""
    supplied = Path(model).expanduser()
    if supplied.is_dir():
        supplied = supplied / "model.pt"
    if supplied.is_file():
        return str(supplied.resolve())
    if supplied.is_absolute() or model.startswith((".", "~")) or supplied.suffix.lower() == ".pt":
        raise FileNotFoundError(f"本地 LingBot 权重不存在: {supplied}")
    if model.startswith("robbyant/lingbot-depth-"):
        local = EAI_DIR.parent / "models" / model.split("/", 1)[1] / "model.pt"
        if local.is_file():
            return str(local)
    return model


def _find_rgb(example_dir: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = example_dir / f"rgb{ext}"
        if candidate.is_file():
            return candidate
    return None


def _synthesize_sparse_depth_png(
    out_path: Path,
    *,
    height: int,
    width: int,
    sparse_z_m: float,
    seed_uv: tuple[int, int] | None = None,
    depth_scale: float = 1000.0,
) -> Path:
    """无传感器 depth 时造稀疏先验（毫米 uint16 PNG），供 MDM 补全。"""
    import cv2

    z = float(sparse_z_m)
    if not DEPTH_MIN_M < z < DEPTH_MAX_M:
        raise ValueError("稀疏深度先验必须在 0.08–2.5m 内")
    mm = int(round(z * float(depth_scale)))
    mm = max(1, min(65535, mm))
    depth_u16 = np.zeros((height, width), dtype=np.uint16)
    # 中心 + 种子 + 稀疏网格，避免全零导致 encoder 无有效深度 token
    pts = [(width // 2, height // 2)]
    if seed_uv is not None:
        pts.append((int(seed_uv[0]), int(seed_uv[1])))
    step_u = max(8, width // 8)
    step_v = max(8, height // 8)
    for v in range(step_v // 2, height, step_v):
        for u in range(step_u // 2, width, step_u):
            pts.append((u, v))
    for u, v in pts:
        uu = max(0, min(width - 1, int(u)))
        vv = max(0, min(height - 1, int(v)))
        depth_u16[vv, uu] = mm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), depth_u16):
        raise RuntimeError(f"无法写出稀疏 depth: {out_path}")
    return out_path


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
    depth_raw = (args.depth or "").strip()
    depth: Path | None = (
        Path(os.path.abspath(os.path.expanduser(depth_raw))) if depth_raw else None
    )
    intrinsics = Path(os.path.abspath(os.path.expanduser(args.intrinsics or "")))
    missing = []
    if not rgb.is_file():
        missing.append(f"rgb={rgb}")
    if not intrinsics.is_file():
        missing.append(f"intrinsics={intrinsics}")
    if missing:
        raise FileNotFoundError("输入文件不存在: " + ", ".join(missing))

    if depth is not None:
        if not depth.is_file():
            raise FileNotFoundError(f"指定深度文件不存在: {depth}")
        return rgb, depth, intrinsics

    # 允许无 depth：用稀疏平面先验（LIBERO 等 RGB-only 场景）
    if not bool(getattr(args, "allow_empty_depth", False)):
        raise FileNotFoundError(
            f"depth 不存在: {depth or '(未提供)'}；"
            "若要从 RGB 稀疏先验补深度请加 --allow-empty-depth"
        )
    import cv2

    img = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取 rgb: {rgb}")
    h, w = img.shape[:2]
    seed = None
    if getattr(args, "seed_u", None) is not None and getattr(args, "seed_v", None) is not None:
        seed = (int(args.seed_u), int(args.seed_v))
    out_dir = Path(os.path.abspath(os.path.expanduser(args.out)))
    synth = out_dir / "raw_depth_sparse.png"
    _synthesize_sparse_depth_png(
        synth,
        height=h,
        width=w,
        sparse_z_m=float(getattr(args, "sparse_z", 0.45)),
        seed_uv=seed,
        depth_scale=float(args.depth_scale),
    )
    print(
        f"[lingbot_depth] 无传感器 depth → 稀疏先验 {synth} "
        f"z={float(getattr(args, 'sparse_z', 0.45)):.3f}m",
        flush=True,
    )
    return rgb, synth, intrinsics


def main() -> int:
    parser = argparse.ArgumentParser(description="LingBot-Depth worker")
    parser.add_argument("--example", default="", help="examples/<id> under lingbot-depth")
    parser.add_argument("--rgb", default="", help="RGB image path")
    parser.add_argument("--depth", default="", help="uint16 PNG 深度（默认毫米）或浮点 NPY（米）；可空")
    parser.add_argument("--intrinsics", default="", help="3x3 像素内参 txt/json，对应输入深度的分辨率")
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
    parser.add_argument(
        "--allow-empty-depth",
        action="store_true",
        help="无 depth 文件时用 --sparse-z 造稀疏先验再补全",
    )
    parser.add_argument(
        "--sparse-z",
        type=float,
        default=0.45,
        help="allow-empty-depth 时稀疏先验深度（米）",
    )
    parser.add_argument("--seed-u", type=int, default=None)
    parser.add_argument("--seed-v", type=int, default=None)
    args = parser.parse_args()

    cache_dir = (args.cache_dir or "").strip() or str(EAI_DIR / ".cache" / "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir


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
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"无法读取 RGB: {rgb_path}")
        sensor_depth_np = load_depth_input(depth_path, scale=float(args.depth_scale))
        image_bgr, sensor_depth_np = align_rgb_to_depth(image_bgr, sensor_depth_np)
        image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = sensor_depth_np.shape
        depth_np = sanitize_depth(sensor_depth_np)
        if not np.any(depth_np > 0):
            raise ValueError("输入深度在 0.08–2.5m 范围内没有有效测量")
        image_tensor = torch.as_tensor(image_np, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        depth_tensor = torch.as_tensor(depth_np, dtype=torch.float32, device=device)

        print(
            f"[lingbot_depth] image={w}x{h} "
            f"depth_range="
            f"{(depth_np[depth_np > 0].min() if np.any(depth_np > 0) else 0):.3f}"
            f"-{depth_np.max():.3f}m",
            flush=True,
        )

        intrinsics = example.load_intrinsics(str(intrinsics_path), w, h)
        pixel_k = np.asarray(intrinsics, dtype=np.float64).copy()
        pixel_k[0] *= w
        pixel_k[1] *= h
        validate_intrinsics(pixel_k)
        intrinsics_tensor = torch.tensor(
            intrinsics, dtype=torch.float32, device=device
        ).unsqueeze(0)

        print(f"[lingbot_depth] loading model…", flush=True)
        t0 = time.time()
        checkpoint = _resolve_model_checkpoint(args.model)
        print(f"[lingbot_depth] checkpoint={checkpoint}", flush=True)
        model = MDMModel.from_pretrained(checkpoint).to(device).eval()
        model.enable_pytorch_native_sdpa()
        load_s = time.time() - t0
        print(f"[lingbot_depth] model loaded in {load_s:.2f}s", flush=True)

        print(f"[lingbot_depth] inference…", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()
        with torch.no_grad():
            output = model.infer(
                image_tensor,
                depth_in=depth_tensor,
                apply_mask=not args.no_mask,
                intrinsics=intrinsics_tensor,
                use_fp16=device.type == "cuda",
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_s = time.time() - t1

        prediction = output["depth"].detach().cpu().numpy()
        if prediction.shape == (1, h, w):
            prediction = prediction[0]
        depth_pred, points_pred, valid_mask = prediction_to_cloud(prediction, pixel_k, (h, w))
        print(
            f"[lingbot_depth] inference {infer_s:.3f}s "
            f"refined="
            f"{(depth_pred[depth_pred > 0].min() if np.any(depth_pred > 0) else 0):.3f}"
            f"-{depth_pred.max():.3f}m",
            flush=True,
        )

        np.save(out_dir / "depth_sensor.npy", sensor_depth_np)
        np.save(out_dir / "depth_input.npy", depth_np)
        np.save(out_dir / "depth_refined.npy", depth_pred)

        depth_raw_color = example.depth_to_color_opencv(depth_np, vmin=DEPTH_MIN_M, vmax=DEPTH_MAX_M)
        depth_pred_color = example.depth_to_color_opencv(depth_pred, vmin=DEPTH_MIN_M, vmax=DEPTH_MAX_M)
        depth_concat = np.concatenate([depth_raw_color, depth_pred_color], axis=1)

        rgb_out = out_dir / "rgb.png"
        depth_in_out = out_dir / "depth_input.png"
        depth_ref_out = out_dir / "depth_refined.png"
        cmp_out = out_dir / "depth_comparison.png"
        ply_out = out_dir / "point_cloud.ply"

        for path, array in (
            (rgb_out, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)),
            (depth_in_out, depth_raw_color), (depth_ref_out, depth_pred_color),
            (cmp_out, depth_concat), (out_dir / "mask_valid.png", valid_mask.astype(np.uint8) * 255),
        ):
            if not cv2.imwrite(str(path), array):
                raise OSError(f"无法写出结果图像: {path}")

        verts = points_pred[valid_mask][::2]
        verts_color = image_np[valid_mask][::2]
        point_cloud = trimesh.PointCloud(verts, verts_color)
        point_cloud.export(ply_out)

        result = {
            "ok": True,
            "model": args.model,
            "device": str(device),
            "load_s": round(load_s, 3),
            "infer_s": round(infer_s, 3),
            "width": int(w),
            "height": int(h),
            "points": int(len(verts)),
            "depth_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
            "input_valid_pixels": int(np.count_nonzero(valid_depth_mask(depth_np))),
            "output_valid_pixels": int(np.count_nonzero(valid_mask)),
            "mask_applied": not args.no_mask,
            "intrinsics_pixels": pixel_k.tolist(),
            "depth_units": "meters",
            "input_source": "sparse_prior" if depth_path.name == "raw_depth_sparse.png" else "sensor_or_file",
            "valid_mask": str(out_dir / "mask_valid.png"),
            "rgb": str(rgb_out),
            "depth_input": str(depth_in_out),
            "depth_refined": str(depth_ref_out),
            "comparison": str(cmp_out),
            "point_cloud": str(ply_out),
            "out_dir": str(out_dir),
        }
        common = valid_depth_mask(depth_np) & valid_mask
        if np.any(common):
            change_mm = np.abs(depth_pred[common] - depth_np[common]) * 1000.0
            result["median_change_mm"] = float(np.median(change_mm))
            result["p95_change_mm"] = float(np.percentile(change_mm, 95))
        (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _emit_result(result)
        return 0
    except Exception as exc:
        import traceback

        traceback.print_exc()
        _emit_result({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
