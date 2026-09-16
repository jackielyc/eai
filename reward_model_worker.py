#!/usr/bin/env python3
"""Score images with RLinf embodied reward models (single / pair / batch)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

EAI_DIR = Path(__file__).resolve().parent
RLINF_ROOT = Path(
    os.environ.get(
        "RLINF_ROOT",
        "/share_data/projects/mahjong/share/personal/liyichao/RLinf",
    )
).expanduser()


def _emit_result(payload: dict) -> None:
    print("[RESULT] " + json.dumps(payload, ensure_ascii=False), flush=True)


def _ensure_rlinf(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "rlinf" / "models" / "embodiment" / "reward").is_dir():
        raise FileNotFoundError(f"RLinf reward 包不存在: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _load_image_rgb(path: str) -> np.ndarray:
    import cv2

    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"无法读取图像: {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _parse_image_size(raw: str) -> list[int]:
    text = (raw or "3,224,224").strip()
    parts = [int(x.strip()) for x in text.replace("x", ",").split(",") if x.strip()]
    if len(parts) == 1:
        s = parts[0]
        return [3, s, s]
    if len(parts) == 2:
        return [3, parts[0], parts[1]]
    if len(parts) >= 3:
        return [parts[0], parts[1], parts[2]]
    return [3, 224, 224]


def _build_model(args: argparse.Namespace):
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.reward import get_reward_model_class

    image_size = _parse_image_size(args.image_size)
    cfg_dict: dict[str, Any] = {
        "model_type": args.model_type,
        "model_path": args.ckpt or None,
        "arch": args.arch,
        "pretrained": bool(args.ckpt is None or args.ckpt == ""),
        "hidden_dim": args.hidden_dim,
        "dropout": 0.0,
        "image_size": image_size,
        "normalize": True,
        "precision": args.precision,
        "reward_threshold": args.threshold if args.threshold >= 0 else None,
        "add_value_head": False,
        "is_lora": False,
        "freeze_vit": False,
        "use_flash_attention": False,
    }
    if args.model_type == "vlm":
        if not args.ckpt:
            raise ValueError("vlm 模式必须提供 --ckpt（HF 模型目录）")
        cfg_dict.update(
            {
                "pretrained": False,
                "max_new_tokens": 32,
                "do_sample": False,
                "temperature": 0.0,
                "input_builder_name": "base_vlm_input_builder",
                "input_builder_params": {},
                "reward_parser_name": "base_reward_parser",
                "reward_parser_params": {},
            }
        )
    cfg = OmegaConf.create(cfg_dict)
    cls = get_reward_model_class(args.model_type)
    model = cls(cfg)
    model.eval()
    return model


def _resolve_device(device: str):
    import torch

    d = (device or "auto").strip().lower()
    if d in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _score_images(
    model,
    images_rgb: list[np.ndarray],
    *,
    device,
    task: str = "",
    model_type: str = "resnet",
) -> list[float]:
    import torch

    batch = np.stack(images_rgb, axis=0)  # NHWC uint8
    if model_type == "vlm":
        observations = {
            "main_images": batch,
            "task_descriptions": [task or ""] * len(images_rgb),
        }
        with torch.no_grad():
            rewards = model.compute_reward(observations)
        return [float(x) for x in rewards.detach().cpu().flatten().tolist()]

    # resnet path
    observations = {"main_images": batch}
    param = next(model.parameters())
    model.to(device=device, dtype=param.dtype)
    with torch.no_grad():
        rewards = model.compute_reward(observations)
    return [float(x) for x in rewards.detach().cpu().flatten().tolist()]


def _run_single(args: argparse.Namespace, model, device) -> dict:
    img = _load_image_rgb(args.image)
    t0 = time.time()
    scores = _score_images(
        model,
        [img],
        device=device,
        task=args.task,
        model_type=args.model_type,
    )
    return {
        "ok": True,
        "mode": "single",
        "score": scores[0],
        "scores": scores,
        "image": os.path.abspath(args.image),
        "infer_s": round(time.time() - t0, 3),
        "device": str(device),
        "model_type": args.model_type,
        "ckpt": args.ckpt,
    }


def _run_pair(args: argparse.Namespace, model, device) -> dict:
    img_a = _load_image_rgb(args.image_a)
    img_b = _load_image_rgb(args.image_b)
    t0 = time.time()
    scores = _score_images(
        model,
        [img_a, img_b],
        device=device,
        task=args.task,
        model_type=args.model_type,
    )
    score_a, score_b = scores[0], scores[1]
    prefer = "A" if score_a >= score_b else "B"
    return {
        "ok": True,
        "mode": "pair",
        "score_a": score_a,
        "score_b": score_b,
        "delta": score_a - score_b,
        "prefer": prefer,
        "image_a": os.path.abspath(args.image_a),
        "image_b": os.path.abspath(args.image_b),
        "infer_s": round(time.time() - t0, 3),
        "device": str(device),
        "model_type": args.model_type,
        "ckpt": args.ckpt,
    }


def _run_batch(args: argparse.Namespace, model, device) -> dict:
    import torch

    from rlinf.data.datasets.reward_model import RewardDatasetPayload

    path = os.path.abspath(os.path.expanduser(args.dataset))
    payload = RewardDatasetPayload.load(path)
    n = len(payload.images)
    max_n = int(args.max_samples) if args.max_samples and args.max_samples > 0 else n
    max_n = min(max_n, n)
    if max_n <= 0:
        raise ValueError("数据集为空")

    threshold = float(args.threshold) if args.threshold >= 0 else 0.5
    preds: list[int] = []
    scores: list[float] = []
    labels: list[int] = []
    t0 = time.time()
    bs = max(1, int(args.batch_size))

    for start in range(0, max_n, bs):
        end = min(start + bs, max_n)
        chunk_imgs: list[np.ndarray] = []
        for i in range(start, end):
            img = payload.images[i]
            if isinstance(img, torch.Tensor):
                arr = img.detach().cpu().numpy()
            else:
                arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            chunk_imgs.append(arr)
            labels.append(int(payload.labels[i]))
        chunk_scores = _score_images(
            model,
            chunk_imgs,
            device=device,
            task=args.task,
            model_type=args.model_type,
        )
        scores.extend(chunk_scores)
        preds.extend([1 if s >= threshold else 0 for s in chunk_scores])
        print(
            f"[batch] {end}/{max_n}  last_score={chunk_scores[-1]:.4f}",
            flush=True,
        )

    correct = sum(int(p == y) for p, y in zip(preds, labels))
    accuracy = correct / max_n
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    pos_scores = [s for s, y in zip(scores, labels) if y == 1]
    neg_scores = [s for s, y in zip(scores, labels) if y == 0]

    def _mean(xs: list[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    return {
        "ok": True,
        "mode": "batch",
        "dataset": path,
        "num_samples": max_n,
        "num_total": n,
        "threshold": threshold,
        "accuracy": accuracy,
        "correct": correct,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_score": _mean(scores),
        "pos_mean": _mean(pos_scores),
        "neg_mean": _mean(neg_scores),
        "infer_s": round(time.time() - t0, 3),
        "device": str(device),
        "model_type": args.model_type,
        "ckpt": args.ckpt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RLinf reward model worker")
    parser.add_argument(
        "--mode",
        choices=("single", "pair", "batch"),
        required=True,
    )
    parser.add_argument("--rlinf-root", default=str(RLINF_ROOT))
    parser.add_argument("--ckpt", default="", help="checkpoint .pt/.pth/.safetensors or HF dir")
    parser.add_argument("--model-type", default="resnet", choices=("resnet", "vlm", "history_vlm"))
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--image-size", default="3,224,224")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="batch 二分类阈值；single/pair 传给 model.cfg（<0 表示不截断）",
    )
    parser.add_argument("--task", default="", help="VLM task description")
    parser.add_argument("--image", default="", help="single mode image")
    parser.add_argument("--image-a", default="", help="pair mode image A")
    parser.add_argument("--image-b", default="", help="pair mode image B")
    parser.add_argument("--dataset", default="", help="batch mode .pt path")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        root = Path(args.rlinf_root or RLINF_ROOT)
        print(f"[reward] rlinf_root={root}", flush=True)
        _ensure_rlinf(root)
        if not args.ckpt.strip():
            raise ValueError("必须提供 --ckpt（reward 权重路径）")
        args.ckpt = os.path.abspath(os.path.expanduser(args.ckpt.strip()))
        if not (os.path.isfile(args.ckpt) or os.path.isdir(args.ckpt)):
            raise FileNotFoundError(f"ckpt 不存在: {args.ckpt}")

        t_load = time.time()
        print(
            f"[reward] loading model_type={args.model_type} arch={args.arch} ckpt={args.ckpt}",
            flush=True,
        )
        model = _build_model(args)
        device = _resolve_device(args.device)
        print(f"[reward] loaded in {time.time() - t_load:.1f}s device={device}", flush=True)

        if args.mode == "single":
            if not args.image:
                raise ValueError("--image 必填")
            result = _run_single(args, model, device)
        elif args.mode == "pair":
            if not args.image_a or not args.image_b:
                raise ValueError("--image-a 与 --image-b 必填")
            result = _run_pair(args, model, device)
        else:
            if not args.dataset:
                raise ValueError("--dataset 必填")
            result = _run_batch(args, model, device)
        _emit_result(result)
        return 0
    except Exception as exc:
        _emit_result({"ok": False, "error": str(exc), "mode": args.mode})
        print(f"[ERROR] {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
