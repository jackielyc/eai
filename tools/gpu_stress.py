#!/usr/bin/env python3
"""GPU compute stress test — saturate SM utilization across one or more GPUs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class _GpuSnapshot:
    physical_index: int
    util_gpu: float
    mem_used_mib: float
    mem_total_mib: float
    compute_procs: int

    @property
    def mem_used_frac(self) -> float:
        if self.mem_total_mib <= 0:
            return 1.0
        return self.mem_used_mib / self.mem_total_mib

    def is_idle(
        self,
        *,
        util_max: float,
        mem_mib_max: float,
        mem_frac_max: float,
        allow_compute_procs: bool,
    ) -> bool:
        if not allow_compute_procs and self.compute_procs > 0:
            return False
        if self.util_gpu > util_max:
            return False
        if self.mem_used_mib > mem_mib_max and self.mem_used_frac > mem_frac_max:
            return False
        return True


def _run_nvidia_smi(args: List[str]) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("nvidia-smi unavailable") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "nvidia-smi failed").strip()
        raise RuntimeError(err)
    return proc.stdout


def _query_gpu_snapshots() -> dict[int, _GpuSnapshot]:
    text = _run_nvidia_smi(
        [
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    snapshots: dict[int, _GpuSnapshot] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        util = float(parts[1]) if parts[1] not in ("[N/A]", "N/A", "") else 0.0
        mem_used = float(parts[2]) if parts[2] not in ("[N/A]", "N/A", "") else 0.0
        mem_total = float(parts[3]) if parts[3] not in ("[N/A]", "N/A", "") else 0.0
        snapshots[idx] = _GpuSnapshot(idx, util, mem_used, mem_total, 0)

    try:
        proc_text = _run_nvidia_smi(
            ["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
        )
        uuid_text = _run_nvidia_smi(
            ["--query-gpu=index,uuid", "--format=csv,noheader"]
        )
    except RuntimeError:
        return snapshots

    uuid_to_index: dict[str, int] = {}
    for line in uuid_text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, uuid = [p.strip() for p in line.split(",", 1)]
        uuid_to_index[uuid] = int(idx_str)

    proc_counts = {idx: 0 for idx in snapshots}
    for line in proc_text.splitlines():
        line = line.strip()
        if not line:
            continue
        uuid, pid = [p.strip() for p in line.split(",", 1)]
        if not pid or pid in ("[N/A]", "N/A"):
            continue
        idx = uuid_to_index.get(uuid)
        if idx is not None:
            proc_counts[idx] = proc_counts.get(idx, 0) + 1

    for idx, snap in snapshots.items():
        snapshots[idx] = _GpuSnapshot(
            snap.physical_index,
            snap.util_gpu,
            snap.mem_used_mib,
            snap.mem_total_mib,
            proc_counts.get(idx, 0),
        )
    return snapshots


def _visible_physical_indices(device_count: int) -> List[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return list(range(device_count))

    mapping: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            mapping.append(int(part))
            continue
        # UUID / MIG / other selectors: fall back to sequential torch indices.
        return list(range(device_count))

    if len(mapping) != device_count:
        return list(range(device_count))
    return mapping


def _format_gpu_status(snap: _GpuSnapshot) -> str:
    mem_gb = snap.mem_used_mib / 1024.0
    total_gb = snap.mem_total_mib / 1024.0
    proc = f", procs={snap.compute_procs}" if snap.compute_procs else ""
    return (
        f"util={snap.util_gpu:.0f}%, mem={mem_gb:.1f}/{total_gb:.1f} GB"
        f" ({snap.mem_used_frac * 100:.1f}%){proc}"
    )


def _detect_idle_torch_indices(
    device_count: int,
    *,
    util_max: float,
    mem_mib_max: float,
    mem_frac_max: float,
    allow_compute_procs: bool,
) -> tuple[List[int], dict[int, _GpuSnapshot]]:
    mapping = _visible_physical_indices(device_count)
    snapshots = _query_gpu_snapshots()

    idle: List[int] = []
    for torch_idx, physical_idx in enumerate(mapping):
        snap = snapshots.get(physical_idx)
        if snap is None:
            print(
                f"[warn] GPU torch:{torch_idx} physical:{physical_idx} not found in nvidia-smi; treating as busy",
                flush=True,
            )
            continue
        state = "idle" if snap.is_idle(
            util_max=util_max,
            mem_mib_max=mem_mib_max,
            mem_frac_max=mem_frac_max,
            allow_compute_procs=allow_compute_procs,
        ) else "busy"
        label = "physical" if mapping[torch_idx] != torch_idx else "torch"
        print(
            f"[info] GPU {torch_idx} ({label} {physical_idx}): {state} | {_format_gpu_status(snap)}",
            flush=True,
        )
        if state == "idle":
            idle.append(torch_idx)
    return idle, snapshots


def _select_gpus(
    gpu_ids_raw: Optional[str],
    num_gpus: Optional[int],
    *,
    all_gpus: bool,
    idle_util_max: float,
    idle_mem_mib_max: float,
    idle_mem_frac_max: float,
    allow_compute_procs: bool,
) -> List[int]:
    import torch

    total = torch.cuda.device_count()
    if total == 0:
        raise SystemExit("No CUDA GPUs detected.")

    if gpu_ids_raw:
        gpu_ids = [int(x.strip()) for x in gpu_ids_raw.split(",") if x.strip()]
        bad = [i for i in gpu_ids if i < 0 or i >= total]
        if bad:
            raise SystemExit(f"Invalid GPU id(s) {bad}; available: 0..{total - 1}")
        if all_gpus:
            selected = gpu_ids
        else:
            idle_set = set(
                _detect_idle_torch_indices(
                    total,
                    util_max=idle_util_max,
                    mem_mib_max=idle_mem_mib_max,
                    mem_frac_max=idle_mem_frac_max,
                    allow_compute_procs=allow_compute_procs,
                )[0]
            )
            selected = [i for i in gpu_ids if i in idle_set]
            skipped = [i for i in gpu_ids if i not in idle_set]
            if skipped:
                print(f"[info] skipped busy requested GPU(s): {skipped}", flush=True)
        if not selected:
            raise SystemExit("No idle GPUs among the requested --gpu-ids.")
        if num_gpus is not None:
            if num_gpus < 1 or num_gpus > len(selected):
                raise SystemExit(
                    f"--num-gpus must be in [1, {len(selected)}] for selected idle GPUs, got {num_gpus}"
                )
            selected = selected[:num_gpus]
        return selected

    if all_gpus:
        selected = list(range(total))
    else:
        selected = _detect_idle_torch_indices(
            total,
            util_max=idle_util_max,
            mem_mib_max=idle_mem_mib_max,
            mem_frac_max=idle_mem_frac_max,
            allow_compute_procs=allow_compute_procs,
        )[0]
        if not selected:
            raise SystemExit(
                "No idle GPUs detected among visible devices. "
                "Use --all-gpus to stress every visible GPU, or free GPUs first."
            )

    if num_gpus is not None:
        if num_gpus < 1 or num_gpus > len(selected):
            raise SystemExit(
                f"--num-gpus must be in [1, {len(selected)}] for selected GPUs, got {num_gpus}"
            )
        selected = selected[:num_gpus]
    return selected


def _parse_host_items(items: List[str]) -> List[str]:
    hosts: List[str] = []
    seen: set[str] = set()
    for item in items:
        for part in item.replace(";", ",").split(","):
            host = part.strip()
            if not host or host.startswith("#") or host in seen:
                continue
            seen.add(host)
            hosts.append(host)
    return hosts


def _load_hosts_file(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return _parse_host_items(lines)


def _collect_hosts(args: argparse.Namespace) -> List[str]:
    hosts: List[str] = []
    if args.hosts:
        hosts.extend(_parse_host_items(args.hosts))
    if args.hosts_file:
        hosts.extend(_load_hosts_file(Path(args.hosts_file)))
    return _parse_host_items(hosts)


def _is_local_host(host: str) -> bool:
    name = host.split("@")[-1]
    name = name.split("%")[0]
    if ":" in name and not name.startswith("["):
        name = name.rsplit(":", 1)[0]
    name = name.strip("[]").lower()
    if name in {"localhost", "127.0.0.1", "::1"}:
        return True
    local_names = {
        socket.gethostname().lower(),
        socket.getfqdn().lower(),
        socket.gethostname().split(".")[0].lower(),
    }
    return name in local_names or name.split(".")[0] in local_names


def _argv_without_hosts(argv: List[str]) -> List[str]:
    single_flags = {"--hosts-file", "--ssh-user", "--ssh-opts"}
    multi_flags = {"--hosts"}
    out: List[str] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if any(arg.startswith(f"{flag}=") for flag in single_flags | multi_flags):
            i += 1
            continue
        if arg in single_flags:
            i += 2
            continue
        if arg in multi_flags:
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                i += 1
            continue
        out.append(arg)
        i += 1
    return out


def _ssh_target(host: str, ssh_user: Optional[str]) -> str:
    if "@" in host or not ssh_user:
        return host
    return f"{ssh_user}@{host}"


def _prefix_stream(proc: subprocess.Popen[str], host: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{host}] {line}", end="" if line.endswith("\n") else "\n", flush=True)


def _stop_process(proc: subprocess.Popen[str], *, grace: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    pid = proc.pid
    for sig in (signal.SIGINT, signal.SIGTERM):
        if proc.poll() is not None:
            return
        try:
            os.killpg(pid, sig)
        except OSError:
            try:
                proc.send_signal(sig)
            except OSError:
                return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
    if proc.poll() is None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait(timeout=5)


def _run_on_hosts(hosts: List[str], args: argparse.Namespace) -> None:
    forwarded = _argv_without_hosts(sys.argv)
    python = sys.executable
    script = str(Path(__file__).resolve())
    remote_cmd = ["exec", python, script, *forwarded]
    ssh_opts = shlex.split(args.ssh_opts) if args.ssh_opts else []

    print(f"[info] remote hosts={hosts}", flush=True)
    print(f"[info] remote cmd={shlex.join(remote_cmd[1:])}", flush=True)

    procs: List[tuple[str, subprocess.Popen[str]]] = []
    pumps: List[threading.Thread] = []
    stop = threading.Event()

    def _handle_sig(signum, _frame):
        print(f"\n[info] signal {signum}, stopping remote jobs...", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_sig)

    try:
        for host in hosts:
            if _is_local_host(host):
                cmd = [python, script, *forwarded]
                print(f"[info] local {host}: {shlex.join(cmd)}", flush=True)
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            else:
                target = _ssh_target(host, args.ssh_user)
                cmd = [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                    *ssh_opts,
                    target,
                    shlex.join(remote_cmd),
                ]
                print(f"[info] ssh {target}", flush=True)
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
            procs.append((host, proc))
            pump = threading.Thread(target=_prefix_stream, args=(proc, host), daemon=True)
            pump.start()
            pumps.append(pump)

        deadline = time.time() + args.duration if args.duration > 0 else None
        while not stop.is_set():
            alive = [(host, proc) for host, proc in procs if proc.poll() is None]
            if not alive:
                break
            if deadline is not None and time.time() >= deadline:
                print("[info] duration reached, stopping remote jobs...", flush=True)
                break
            time.sleep(0.3)
    finally:
        for host, proc in procs:
            if proc.poll() is None:
                print(f"[info] stopping {host} pid={proc.pid}", flush=True)
            _stop_process(proc)
        for pump in pumps:
            pump.join(timeout=2)

    failed = [
        host
        for host, proc in procs
        if proc.returncode not in (0, None, -signal.SIGINT, -signal.SIGTERM, -signal.SIGHUP, -signal.SIGKILL)
    ]
    codes = {host: proc.returncode for host, proc in procs}
    print(f"[info] remote exit codes: {codes}", flush=True)
    if failed:
        raise SystemExit(f"remote host(s) failed: {failed}")


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
        help="Use at most N selected GPUs (default: all idle visible GPUs)",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU indices, e.g. 0,2,4 (overrides idle auto-pick)",
    )
    parser.add_argument(
        "--all-gpus",
        action="store_true",
        help="Use all visible GPUs instead of auto-detecting idle ones",
    )
    parser.add_argument(
        "--idle-util-max",
        type=float,
        default=5.0,
        help="Max GPU utilization %% to count as idle (via nvidia-smi)",
    )
    parser.add_argument(
        "--idle-mem-mib",
        type=float,
        default=512.0,
        help="Max used memory (MiB) to count as idle when mem fraction is also low",
    )
    parser.add_argument(
        "--idle-mem-frac-max",
        type=float,
        default=0.05,
        help="Max used/total memory fraction to count as idle",
    )
    parser.add_argument(
        "--allow-compute-procs",
        action="store_true",
        help="Treat GPUs with existing compute processes as idle candidates",
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
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=None,
        help="Remote hosts, e.g. gpu-a gpu-b or a,b (ssh; local hostname runs locally)",
    )
    parser.add_argument(
        "--hosts-file",
        default=None,
        help="File with one host per line (# comments allowed)",
    )
    parser.add_argument(
        "--ssh-user",
        default=None,
        help="SSH user if hosts do not contain user@host",
    )
    parser.add_argument(
        "--ssh-opts",
        default="",
        help="Extra ssh options as a single string, e.g. '-p 2222 -i /path/key'",
    )
    args = parser.parse_args()

    hosts = _collect_hosts(args)
    if hosts:
        _run_on_hosts(hosts, args)
        return

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Example:\n"
            "  PYTHON=/path/to/miniconda3/envs/Qwen2.5-VL/bin/python tools/gpu_stress.py"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this PyTorch build.")

    gpu_ids = _select_gpus(
        args.gpu_ids,
        args.num_gpus,
        all_gpus=args.all_gpus,
        idle_util_max=args.idle_util_max,
        idle_mem_mib_max=args.idle_mem_mib,
        idle_mem_frac_max=args.idle_mem_frac_max,
        allow_compute_procs=args.allow_compute_procs,
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    mode = "all visible" if args.all_gpus else "idle auto-detect"
    print(f"[info] visible={visible} mode={mode} using GPUs: {gpu_ids}", flush=True)
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
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_sig)

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
