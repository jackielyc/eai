"""采集本机 CPU / 内存 / GPU 快照（对齐 htop / nvtop 字段）。"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GpuSnapshot:
    index: int
    name: str
    util_pct: float
    mem_util_pct: float
    mem_used_mib: float
    mem_total_mib: float
    mem_free_mib: float
    temp_c: float
    power_w: float
    power_limit_w: float
    graphics_clock_mhz: float = 0.0
    graphics_clock_max_mhz: float = 0.0
    mem_clock_mhz: float = 0.0
    mem_clock_max_mhz: float = 0.0
    fan_pct: float = 0.0
    pcie_gen: int = 0
    pcie_width: int = 0


@dataclass
class GpuProcess:
    gpu_index: int
    pid: int
    name: str
    cmdline: str
    username: str
    mem_used_mib: float
    cpu_pct: float = 0.0
    mem_pct: float = 0.0


@dataclass
class HostProcess:
    pid: int
    username: str
    name: str
    cmdline: str
    status: str
    cpu_pct: float
    mem_pct: float
    rss_b: int
    nice: int
    num_threads: int


@dataclass
class HostSnapshot:
    ts: float
    hostname: str
    cpu_model: str
    cpu_count_logical: int
    cpu_count_physical: int
    cpu_pct: float
    cpu_per_core: List[float]
    load_avg: Tuple[float, float, float]
    uptime_s: float
    # htop-style memory breakdown
    mem_total_b: int
    mem_used_b: int
    mem_available_b: int
    mem_free_b: int
    mem_buffers_b: int
    mem_cached_b: int
    mem_shared_b: int
    mem_pct: float
    swap_total_b: int
    swap_used_b: int
    swap_pct: float
    # tasks
    n_processes: int = 0
    n_threads: int = 0
    n_running: int = 0
    gpus: List[GpuSnapshot] = field(default_factory=list)
    gpu_processes: List[GpuProcess] = field(default_factory=list)
    top_processes: List[HostProcess] = field(default_factory=list)
    error: str = ""


_cpu_model_cache: Optional[str] = None
_psutil_seeded = False
_proc_cpu_seeded = False


def _cpu_model() -> str:
    global _cpu_model_cache
    if _cpu_model_cache is not None:
        return _cpu_model_cache
    model = ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[-1].strip()
                    break
    except OSError:
        pass
    if not model:
        try:
            import platform

            model = platform.processor() or "CPU"
        except Exception:
            model = "CPU"
    _cpu_model_cache = model
    return model


def _parse_float(text: str, default: float = 0.0) -> float:
    raw = (text or "").strip().replace(",", "")
    if not raw or raw in ("[N/A]", "N/A", "None"):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int(text: str, default: int = 0) -> int:
    return int(_parse_float(text, float(default)))


def _query_gpus() -> Tuple[List[GpuSnapshot], str]:
    gpus: List[GpuSnapshot] = []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,memory.free,temperature.gpu,"
                "power.draw,power.limit,"
                "clocks.current.graphics,clocks.max.graphics,"
                "clocks.current.memory,clocks.max.memory,"
                "fan.speed,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        )
    except FileNotFoundError:
        return [], "未找到 nvidia-smi"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return [], f"nvidia-smi 失败: {exc}"

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            continue
        try:
            idx = int(_parse_float(parts[0]))
        except ValueError:
            continue
        mem_used = _parse_float(parts[4])
        mem_total = _parse_float(parts[5])
        mem_free = _parse_float(parts[6]) if len(parts) > 6 else max(0.0, mem_total - mem_used)
        mem_util = _parse_float(parts[3])
        if mem_total > 0 and mem_util <= 0:
            mem_util = 100.0 * mem_used / mem_total
        gpus.append(
            GpuSnapshot(
                index=idx,
                name=parts[1],
                util_pct=_parse_float(parts[2]),
                mem_util_pct=mem_util,
                mem_used_mib=mem_used,
                mem_total_mib=mem_total,
                mem_free_mib=mem_free,
                temp_c=_parse_float(parts[7]),
                power_w=_parse_float(parts[8]),
                power_limit_w=_parse_float(parts[9]),
                graphics_clock_mhz=_parse_float(parts[10]) if len(parts) > 10 else 0.0,
                graphics_clock_max_mhz=_parse_float(parts[11]) if len(parts) > 11 else 0.0,
                mem_clock_mhz=_parse_float(parts[12]) if len(parts) > 12 else 0.0,
                mem_clock_max_mhz=_parse_float(parts[13]) if len(parts) > 13 else 0.0,
                fan_pct=_parse_float(parts[14]) if len(parts) > 14 else 0.0,
                pcie_gen=_parse_int(parts[15]) if len(parts) > 15 else 0,
                pcie_width=_parse_int(parts[16]) if len(parts) > 16 else 0,
            )
        )
    return gpus, ""


def _query_gpu_processes(uuid_to_idx: Dict[str, int]) -> List[GpuProcess]:
    procs: List[GpuProcess] = []
    try:
        pout = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        )
    except Exception:
        return procs

    import psutil

    for line in pout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        uuid, pid_s, name, mem_s = parts[0], parts[1], parts[2], parts[3]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        username = "?"
        cmdline = name
        cpu_pct = 0.0
        mem_pct = 0.0
        try:
            p = psutil.Process(pid)
            with p.oneshot():
                username = p.username() or "?"
                cmd = p.cmdline()
                cmdline = " ".join(cmd) if cmd else (p.name() or name)
                cpu_pct = float(p.cpu_percent(interval=None) or 0.0)
                mem_pct = float(p.memory_percent() or 0.0)
        except (psutil.Error, OSError):
            pass
        procs.append(
            GpuProcess(
                gpu_index=int(uuid_to_idx.get(uuid, -1)),
                pid=pid,
                name=os.path.basename(name) or name,
                cmdline=cmdline,
                username=username,
                mem_used_mib=_parse_float(mem_s),
                cpu_pct=cpu_pct,
                mem_pct=mem_pct,
            )
        )
    procs.sort(key=lambda x: (-x.mem_used_mib, -x.cpu_pct))
    return procs


def _gpu_uuid_map() -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    try:
        uuid_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        )
    except Exception:
        return mapping
    for line in uuid_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                mapping[parts[1]] = int(parts[0])
            except ValueError:
                pass
    return mapping


def _collect_top_processes(limit: int = 40) -> Tuple[List[HostProcess], int, int, int]:
    """返回 (top by CPU, n_processes, n_threads, n_running)。"""
    global _proc_cpu_seeded
    import psutil

    attrs = (
        "pid",
        "username",
        "name",
        "cmdline",
        "status",
        "cpu_percent",
        "memory_percent",
        "memory_info",
        "nice",
        "num_threads",
    )
    rows: List[HostProcess] = []
    n_proc = 0
    n_thr = 0
    n_run = 0
    # First pass seeds cpu_percent; second pass (next call) is meaningful.
    for p in psutil.process_iter(attrs):
        try:
            info = p.info
        except (psutil.Error, OSError):
            continue
        n_proc += 1
        thr = int(info.get("num_threads") or 1)
        n_thr += thr
        st = str(info.get("status") or "")
        if st == psutil.STATUS_RUNNING:
            n_run += 1
        if not _proc_cpu_seeded:
            continue
        cmd = info.get("cmdline") or []
        cmdline = " ".join(cmd) if cmd else (info.get("name") or "")
        mi = info.get("memory_info")
        rss = int(getattr(mi, "rss", 0) or 0)
        rows.append(
            HostProcess(
                pid=int(info.get("pid") or 0),
                username=str(info.get("username") or "?"),
                name=str(info.get("name") or ""),
                cmdline=cmdline,
                status=st[:1].upper() if st else "?",
                cpu_pct=float(info.get("cpu_percent") or 0.0),
                mem_pct=float(info.get("memory_percent") or 0.0),
                rss_b=rss,
                nice=int(info.get("nice") or 0),
                num_threads=thr,
            )
        )
    _proc_cpu_seeded = True
    rows.sort(key=lambda r: (-r.cpu_pct, -r.mem_pct))
    return rows[: max(1, limit)], n_proc, n_thr, n_run


def collect_host_snapshot(*, top_n: int = 40) -> HostSnapshot:
    """采集一次主机资源快照（非阻塞；首次 CPU%/进程% 可能偏低）。"""
    global _psutil_seeded
    import psutil

    if not _psutil_seeded:
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        _psutil_seeded = True

    cpu_pct = float(psutil.cpu_percent(interval=None) or 0.0)
    per_core = [float(x or 0.0) for x in psutil.cpu_percent(interval=None, percpu=True)]
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    try:
        load = os.getloadavg()
    except OSError:
        load = (0.0, 0.0, 0.0)
    try:
        uptime_s = max(0.0, time.time() - float(psutil.boot_time()))
    except Exception:
        uptime_s = 0.0

    # htop Mem: used ≈ total - free - buffers - cached (approx)
    buffers = int(getattr(vm, "buffers", 0) or 0)
    cached = int(getattr(vm, "cached", 0) or 0)
    shared = int(getattr(vm, "shared", 0) or 0)
    free_b = int(getattr(vm, "free", 0) or 0)
    # "used" excluding reclaimable cache (htop green part)
    used_no_cache = max(0, int(vm.total) - free_b - buffers - cached)

    gpus, gpu_err = _query_gpus()
    uuid_map = _gpu_uuid_map() if gpus else {}
    gpu_procs = _query_gpu_processes(uuid_map) if gpus else []
    top_procs, n_proc, n_thr, n_run = _collect_top_processes(limit=top_n)

    try:
        host = socket.gethostname()
    except OSError:
        host = "localhost"

    return HostSnapshot(
        ts=time.time(),
        hostname=host,
        cpu_model=_cpu_model(),
        cpu_count_logical=int(psutil.cpu_count(logical=True) or len(per_core) or 0),
        cpu_count_physical=int(psutil.cpu_count(logical=False) or 0),
        cpu_pct=cpu_pct,
        cpu_per_core=per_core,
        load_avg=(float(load[0]), float(load[1]), float(load[2])),
        uptime_s=uptime_s,
        mem_total_b=int(vm.total),
        mem_used_b=used_no_cache,
        mem_available_b=int(vm.available),
        mem_free_b=free_b,
        mem_buffers_b=buffers,
        mem_cached_b=cached,
        mem_shared_b=shared,
        mem_pct=float(vm.percent),
        swap_total_b=int(swap.total),
        swap_used_b=int(swap.used),
        swap_pct=float(swap.percent),
        n_processes=n_proc,
        n_threads=n_thr,
        n_running=n_run,
        gpus=gpus,
        gpu_processes=gpu_procs,
        top_processes=top_procs,
        error=gpu_err,
    )


def format_bytes(n: float) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(x)} {unit}"
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} TiB"


def format_uptime(seconds: float) -> str:
    s = max(0, int(seconds))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours:02d}:{mins:02d}:{secs:02d}"
