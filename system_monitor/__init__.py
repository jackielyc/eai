"""本机 CPU / GPU / 内存资源监控（htop / nvtop 风格）。"""

from system_monitor.metrics import HostSnapshot, collect_host_snapshot

__all__ = ["HostSnapshot", "collect_host_snapshot"]
