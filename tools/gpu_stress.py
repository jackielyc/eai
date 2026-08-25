#!/usr/bin/env python3
"""GPU compute stress test — saturate SM utilization across one or more GPUs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import List, Optional


def _parse_gpu_ids(raw: Optional[str], num_gpus: Optional[int]) -> List[int]:
    import torch

    total = torch.cuda.device_count()
    if total == 0:
        raise SystemExit("No CUDA GPUs detected.")

    if raw:
        ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        bad = [i for i in ids if i < 0 or i >= total]
        if bad:
            raise SystemExit(f"Invalid GPU id(s) {bad}; available: 0..{total - 1}")
        return ids

    count = num_gpus if num_gpus is not None else total
    if count < 1 or count > total:
        raise SystemExit(f"--num-gpus must be in [1, {total}], got {count}")
    return list(range(count))


def _pick_matrix_size(
    gpu_id: int,
    dtype_name: str,
    mem_fraction: float,
    num_streams: int,
) -> int:
    import torch

    torch.cuda.set_device(gpu_id)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_id)
    bytes_per = 2 if dtype_name in ("fp16", "bf16") else 4
    # a, b, and one output buffer per stream, plus ~15% headroom for cuBLAS workspace.
    num_buffers = 2 + num_streams
    budget = int(free_bytes * mem_fraction)
    n = int((budget / (num_buffers * bytes_per)) ** 0.5)
    n = max(2048, min(32768, (n // 256) * 256))
    return n


def _worker(
    gpu_id: int,
    stop_event: mp.Event,
    matrix_size: int,
    dtype_name: str,
    num_streams: int,
    mem_fraction: float,
    report_interval: float,
) -> None:
    import torch

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    if dtype_name == "fp16":
        dtype = torch.float16
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    name = torch.cuda.get_device_name(gpu_id)
    if matrix_size <= 0:
        matrix_size = _pick_matrix_size(gpu_id, dtype_name, mem_fraction, num_streams)

    # Pre-allocate tensors once; reuse buffers in the hot loop.
    # Retry with smaller matrices if the GPU is partially occupied.
    while matrix_size >= 2048:
        try:
            a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
            b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
            outs = [torch.empty_like(a) for _ in range(num_streams)]
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            matrix_size = max(2048, matrix_size // 2)
    else:
        raise RuntimeError(f"GPU {gpu_id} has insufficient free memory for stress test")

    print(
        f"[gpu {gpu_id}] {name} | size={matrix_size} dtype={dtype_name} streams={num_streams}",
        flush=True,
    )

    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    # Warmup
    for i in range(num_streams):
        with torch.cuda.stream(streams[i]):
            outs[i] = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    start = time.time()
    last_report = start
    iters = 0

    while not stop_event.is_set():
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                outs[i] = torch.matmul(a, outs[i])
        iters += num_streams

        now = time.time()
        if report_interval > 0 and now - last_report >= report_interval:
            elapsed = now - start
            print(
                f"[gpu {gpu_id}] {iters / elapsed:.1f} matmul/s | "
                f"mem {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB",
                flush=True,
            )
            last_report = now

    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    print(f"[gpu {gpu_id}] stopped after {elapsed:.1f}s, {iters} matmuls", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress GPUs with continuous GEMM to drive utilization toward 100%.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all visible GPUs)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU indices, e.g. 0,2,4 (overrides --num-gpus)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=0,
        help="Run time in seconds; 0 means until Ctrl+C",
    )
    parser.add_argument(
        "-s",
        "--matrix-size",
        type=int,
        default=0,
        help="GEMM matrix dimension N (0 = auto from GPU memory)",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
        help="Compute dtype (fp16/bf16 use tensor cores on supported GPUs)",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=4,
        help="CUDA streams per GPU for overlapping GEMM",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.85,
        help="Fraction of GPU memory used when auto-sizing matrices",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=10.0,
        help="Seconds between per-GPU throughput logs (0 to disable)",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Example:\n"
            "  PYTHON=/path/to/miniconda3/envs/Qwen2.5-VL/bin/python tools/gpu_stress.py"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch build.")

    gpu_ids = _parse_gpu_ids(args.gpu_ids, args.num_gpus)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    print(f"[info] visible={visible} using GPUs: {gpu_ids}", flush=True)
    if args.duration > 0:
        print(f"[info] duration={args.duration}s", flush=True)
    else:
        print("[info] running until Ctrl+C", flush=True)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    processes: List[mp.Process] = []

    def _handle_sig(signum, _frame):
        print(f"\n[info] signal {signum}, stopping...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    for gid in gpu_ids:
        p = ctx.Process(
            target=_worker,
            args=(
                gid,
                stop_event,
                args.matrix_size,
                args.dtype,
                args.streams,
                args.mem_fraction,
                args.report_interval,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)

    deadline = time.time() + args.duration if args.duration > 0 else None
    try:
        while True:
            if stop_event.is_set():
                break
            if deadline is not None and time.time() >= deadline:
                print("[info] duration reached, stopping...", flush=True)
                stop_event.set()
                break
            if any(not p.is_alive() for p in processes):
                print("[error] a worker exited unexpectedly", file=sys.stderr, flush=True)
                stop_event.set()
                break
            time.sleep(0.5)
    finally:
        stop_event.set()
        for p in processes:
            p.join(timeout=30)

    print("[info] done. Check utilization with: watch -n1 nvidia-smi", flush=True)


if __name__ == "__main__":
    main()
