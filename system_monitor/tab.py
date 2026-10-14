"""本机资源监控：布局对齐 htop（CPU/Mem/进程）与 nvtop（GPU）。"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from system_monitor.metrics import (
    HostSnapshot,
    collect_host_snapshot,
    format_bytes,
    format_uptime,
)
from system_monitor.widgets import HtopCpuMeters, HtopMemBar, NvtopGpuPanel

_TEXT = "#ececec"
_MUTED = "#b0b0b0"
_GREEN = "#50fa7b"
_RED = "#ff8888"
_BG = "#1e1f22"
_BORDER = "#3d4048"
_MONO = "Monospace"
_TABLE_SS = (
    "QTableWidget {"
    f"  background: #1a1b1e; color: {_TEXT}; gridline-color: {_BORDER};"
    f"  border: 1px solid {_BORDER}; font-family: {_MONO}; font-size: 10px; }}"
    "QHeaderView::section {"
    f"  background: #252628; color: {_MUTED}; border: 1px solid {_BORDER};"
    "  padding: 3px 6px; font-weight: bold; }"
    "QTableWidget::item:selected { background: #3d6ea8; }"
)


def _item(text: str, *, align: int = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    it.setTextAlignment(align)
    return it


class SystemMonitorTab(QWidget):
    """htop + nvtop 风格本机监控。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("systemMonitorTab")
        self.setStyleSheet(
            f"#systemMonitorTab {{ background-color: {_BG}; }}"
            "#systemMonitorTab QLabel { background: transparent; }"
            f"#systemMonitorTab QGroupBox {{"
            f"  color: {_TEXT}; border: 1px solid {_BORDER}; border-radius: 4px;"
            f"  margin-top: 8px; padding-top: 6px; }}"
            f"#systemMonitorTab QGroupBox::title {{"
            f"  subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {_MUTED}; }}"
        )
        self._paused = False
        self._gpu_panels: List[NvtopGpuPanel] = []
        self._gpu_wrap_layout: Optional[QVBoxLayout] = None
        self._sort_cpu = True  # True: CPU%, False: MEM%

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(4)

        # ---- header ----
        top = QHBoxLayout()
        top.setSpacing(8)
        self.host_label = QLabel("主机资源 · htop / nvtop")
        self.host_label.setFont(QFont(_MONO, 10, QFont.Bold))
        self.host_label.setStyleSheet(f"color: {_TEXT};")
        top.addWidget(self.host_label, 1)
        self.task_label = QLabel("")
        self.task_label.setFont(QFont(_MONO, 9))
        self.task_label.setStyleSheet(f"color: {_MUTED};")
        top.addWidget(self.task_label)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setFixedWidth(64)
        self.pause_btn.clicked.connect(self._toggle_pause)
        top.addWidget(self.pause_btn)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setFixedWidth(56)
        self.refresh_btn.clicked.connect(self._sample_once)
        top.addWidget(self.refresh_btn)
        root.addLayout(top)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # ---- upper: htop meters + nvtop GPUs ----
        upper = QWidget()
        upper_l = QVBoxLayout(upper)
        upper_l.setContentsMargins(0, 0, 0, 0)
        upper_l.setSpacing(6)

        htop_box = QGroupBox("htop · CPU / Memory")
        htop_l = QVBoxLayout(htop_box)
        htop_l.setSpacing(4)
        self.cpu_meters = HtopCpuMeters()
        htop_l.addWidget(self.cpu_meters, 1)
        self.mem_bar = HtopMemBar()
        htop_l.addWidget(self.mem_bar)
        self.swap_bar = HtopMemBar()
        htop_l.addWidget(self.swap_bar)
        legend = QLabel("Mem 色块:  used(绿)  buffers(黄)  cache(紫)    ·    对齐 htop")
        legend.setStyleSheet(f"color: {_MUTED}; font-size: 10px;")
        htop_l.addWidget(legend)
        upper_l.addWidget(htop_box, 2)

        nvtop_box = QGroupBox("nvtop · GPU")
        nvtop_outer = QVBoxLayout(nvtop_box)
        self._gpu_scroll = QScrollArea()
        self._gpu_scroll.setWidgetResizable(True)
        self._gpu_scroll.setFrameShape(QFrame.NoFrame)
        self._gpu_scroll.setMinimumHeight(160)
        self._gpu_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        gpu_host = QWidget()
        self._gpu_wrap_layout = QVBoxLayout(gpu_host)
        self._gpu_wrap_layout.setContentsMargins(0, 0, 0, 0)
        self._gpu_wrap_layout.setSpacing(6)
        self._gpu_placeholder = QLabel("检测 NVIDIA GPU…")
        self._gpu_placeholder.setAlignment(Qt.AlignCenter)
        self._gpu_placeholder.setStyleSheet(f"color: {_MUTED};")
        self._gpu_wrap_layout.addWidget(self._gpu_placeholder)
        self._gpu_scroll.setWidget(gpu_host)
        nvtop_outer.addWidget(self._gpu_scroll)
        upper_l.addWidget(nvtop_box, 2)

        splitter.addWidget(upper)

        # ---- lower: process tables ----
        lower = QWidget()
        lower_l = QVBoxLayout(lower)
        lower_l.setContentsMargins(0, 0, 0, 0)
        lower_l.setSpacing(4)

        proc_split = QSplitter(Qt.Horizontal)
        proc_split.setChildrenCollapsible(False)

        # GPU processes (nvtop)
        gpu_proc_box = QGroupBox("nvtop · GPU Processes")
        gpu_proc_l = QVBoxLayout(gpu_proc_box)
        self.gpu_table = QTableWidget(0, 7)
        self.gpu_table.setHorizontalHeaderLabels(
            ["GPU", "PID", "USER", "GPU-MEM", "CPU%", "MEM%", "Command"]
        )
        self.gpu_table.setStyleSheet(_TABLE_SS)
        self.gpu_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.gpu_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.gpu_table.verticalHeader().setVisible(False)
        self.gpu_table.setShowGrid(False)
        self.gpu_table.setAlternatingRowColors(True)
        hh = self.gpu_table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        gpu_proc_l.addWidget(self.gpu_table)
        proc_split.addWidget(gpu_proc_box)

        # Host processes (htop)
        cpu_proc_box = QGroupBox("htop · Processes")
        cpu_proc_l = QVBoxLayout(cpu_proc_box)
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("排序"))
        self.sort_cpu_btn = QPushButton("CPU%")
        self.sort_cpu_btn.setCheckable(True)
        self.sort_cpu_btn.setChecked(True)
        self.sort_cpu_btn.clicked.connect(lambda: self._set_sort(True))
        sort_row.addWidget(self.sort_cpu_btn)
        self.sort_mem_btn = QPushButton("MEM%")
        self.sort_mem_btn.setCheckable(True)
        self.sort_mem_btn.clicked.connect(lambda: self._set_sort(False))
        sort_row.addWidget(self.sort_mem_btn)
        sort_row.addStretch(1)
        cpu_proc_l.addLayout(sort_row)
        self.proc_table = QTableWidget(0, 8)
        self.proc_table.setHorizontalHeaderLabels(
            ["PID", "USER", "NI", "RES", "S", "CPU%", "MEM%", "Command"]
        )
        self.proc_table.setStyleSheet(_TABLE_SS)
        self.proc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.proc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.proc_table.verticalHeader().setVisible(False)
        self.proc_table.setShowGrid(False)
        self.proc_table.setAlternatingRowColors(True)
        ph = self.proc_table.horizontalHeader()
        ph.setStretchLastSection(True)
        for col in range(7):
            ph.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        cpu_proc_l.addWidget(self.proc_table)
        proc_split.addWidget(cpu_proc_box)
        proc_split.setStretchFactor(0, 2)
        proc_split.setStretchFactor(1, 3)

        lower_l.addWidget(proc_split)
        splitter.addWidget(lower)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([520, 320])
        root.addWidget(splitter, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._sample_once)
        QTimer.singleShot(50, self._sample_once)
        self._timer.start()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._paused and not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        if self._paused:
            self._timer.stop()

    def _set_sort(self, by_cpu: bool) -> None:
        self._sort_cpu = by_cpu
        self.sort_cpu_btn.setChecked(by_cpu)
        self.sort_mem_btn.setChecked(not by_cpu)
        self._sample_once()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            self._timer.stop()
            self.pause_btn.setText("继续")
            self.task_label.setText("已暂停")
        else:
            self.pause_btn.setText("暂停")
            self._timer.start()
            self._sample_once()

    def _ensure_gpu_panels(self, n: int) -> None:
        assert self._gpu_wrap_layout is not None
        if len(self._gpu_panels) == n and n > 0:
            return
        while self._gpu_wrap_layout.count():
            item = self._gpu_wrap_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._gpu_panels.clear()
        if n <= 0:
            ph = QLabel("未检测到 NVIDIA GPU（需 nvidia-smi）")
            ph.setAlignment(Qt.AlignCenter)
            ph.setStyleSheet(f"color: {_MUTED};")
            self._gpu_wrap_layout.addWidget(ph)
            return
        for _ in range(n):
            panel = NvtopGpuPanel()
            panel.setMinimumHeight(155)
            self._gpu_wrap_layout.addWidget(panel)
            self._gpu_panels.append(panel)
        self._gpu_wrap_layout.addStretch(1)

    def _sample_once(self) -> None:
        try:
            snap = collect_host_snapshot(top_n=50)
        except Exception as exc:
            self.task_label.setText(f"采样失败: {exc}")
            self.task_label.setStyleSheet(f"color: {_RED};")
            return
        self._apply_snapshot(snap)

    def _apply_snapshot(self, snap: HostSnapshot) -> None:
        phys = snap.cpu_count_physical or "?"
        self.host_label.setText(
            f"{snap.hostname}  ·  {snap.cpu_model}  ·  {phys}C/{snap.cpu_count_logical}T"
        )
        load = snap.load_avg
        self.task_label.setStyleSheet(f"color: {_MUTED};")
        self.task_label.setText(
            f"Tasks: {snap.n_processes}, {snap.n_threads} thr; "
            f"{snap.n_running} running   "
            f"Load: {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}   "
            f"Uptime: {format_uptime(snap.uptime_s)}"
            + (f"   · {snap.error}" if snap.error else "")
        )

        # htop meters
        self.cpu_meters.set_cores(snap.cpu_per_core, avg=snap.cpu_pct)
        self.mem_bar.set_mem(
            title="Mem",
            total=float(snap.mem_total_b),
            used=float(snap.mem_used_b),
            buffers=float(snap.mem_buffers_b),
            cached=float(snap.mem_cached_b),
            detail=(
                f"{format_bytes(snap.mem_used_b)}+"
                f"{format_bytes(snap.mem_buffers_b + snap.mem_cached_b)}/"
                f"{format_bytes(snap.mem_total_b)}"
            ),
        )
        self.swap_bar.set_mem(
            title="Swp",
            total=float(snap.swap_total_b) if snap.swap_total_b else 1.0,
            used=float(snap.swap_used_b),
            buffers=0.0,
            cached=0.0,
            detail=(
                f"{format_bytes(snap.swap_used_b)}/{format_bytes(snap.swap_total_b)}"
                if snap.swap_total_b
                else "0/0"
            ),
        )

        # nvtop GPUs
        n_gpu = len(snap.gpus)
        self._ensure_gpu_panels(n_gpu)
        for i, gpu in enumerate(snap.gpus):
            if i >= len(self._gpu_panels):
                break
            mem_pct = (
                100.0 * gpu.mem_used_mib / gpu.mem_total_mib
                if gpu.mem_total_mib > 0
                else gpu.mem_util_pct
            )
            power = (
                f"{gpu.power_w:.0f}/{gpu.power_limit_w:.0f}W"
                if gpu.power_limit_w > 0
                else f"{gpu.power_w:.0f}W"
            )
            clk = (
                f"GCLK {gpu.graphics_clock_mhz:.0f}/{gpu.graphics_clock_max_mhz:.0f}MHz"
                if gpu.graphics_clock_max_mhz > 0
                else f"GCLK {gpu.graphics_clock_mhz:.0f}MHz"
            )
            mclk = (
                f"MCLK {gpu.mem_clock_mhz:.0f}/{gpu.mem_clock_max_mhz:.0f}MHz"
                if gpu.mem_clock_max_mhz > 0
                else f"MCLK {gpu.mem_clock_mhz:.0f}MHz"
            )
            fan = f"FAN {gpu.fan_pct:.0f}%" if gpu.fan_pct > 0 else "FAN N/A"
            pcie = (
                f"PCIe Gen{gpu.pcie_gen}x{gpu.pcie_width}"
                if gpu.pcie_gen and gpu.pcie_width
                else ""
            )
            meta = "  ·  ".join(
                x
                for x in (
                    f"TEMP {gpu.temp_c:.0f}°C",
                    power,
                    fan,
                    clk,
                    mclk,
                    pcie,
                )
                if x
            )
            self._gpu_panels[i].set_stats(
                title=f"Device {gpu.index} [{gpu.name}]",
                gpu_pct=gpu.util_pct,
                mem_pct=mem_pct,
                mem_detail=(
                    f"{gpu.mem_used_mib/1024:.1f}/{gpu.mem_total_mib/1024:.0f}GiB "
                    f"({mem_pct:.0f}%)"
                ),
                meta=meta,
            )

        # GPU process table
        gps = snap.gpu_processes
        self.gpu_table.setRowCount(len(gps))
        for r, proc in enumerate(gps):
            g = str(proc.gpu_index) if proc.gpu_index >= 0 else "?"
            self.gpu_table.setItem(r, 0, _item(g, align=Qt.AlignCenter))
            self.gpu_table.setItem(r, 1, _item(str(proc.pid), align=Qt.AlignRight | Qt.AlignVCenter))
            self.gpu_table.setItem(r, 2, _item(proc.username[:12]))
            self.gpu_table.setItem(
                r,
                3,
                _item(f"{proc.mem_used_mib:.0f}MiB", align=Qt.AlignRight | Qt.AlignVCenter),
            )
            self.gpu_table.setItem(
                r, 4, _item(f"{proc.cpu_pct:.1f}", align=Qt.AlignRight | Qt.AlignVCenter)
            )
            self.gpu_table.setItem(
                r, 5, _item(f"{proc.mem_pct:.1f}", align=Qt.AlignRight | Qt.AlignVCenter)
            )
            self.gpu_table.setItem(r, 6, _item(proc.cmdline or proc.name))
        if not gps:
            self.gpu_table.setRowCount(1)
            self.gpu_table.setItem(0, 6, _item("（无 GPU 计算进程）"))

        # Host process table
        procs = list(snap.top_processes)
        if self._sort_cpu:
            procs.sort(key=lambda p: (-p.cpu_pct, -p.mem_pct))
        else:
            procs.sort(key=lambda p: (-p.mem_pct, -p.cpu_pct))
        self.proc_table.setRowCount(len(procs))
        for r, proc in enumerate(procs):
            self.proc_table.setItem(r, 0, _item(str(proc.pid), align=Qt.AlignRight | Qt.AlignVCenter))
            self.proc_table.setItem(r, 1, _item(proc.username[:12]))
            self.proc_table.setItem(
                r, 2, _item(str(proc.nice), align=Qt.AlignRight | Qt.AlignVCenter)
            )
            self.proc_table.setItem(
                r,
                3,
                _item(format_bytes(proc.rss_b).replace(" ", ""), align=Qt.AlignRight | Qt.AlignVCenter),
            )
            self.proc_table.setItem(r, 4, _item(proc.status, align=Qt.AlignCenter))
            cpu_it = _item(f"{proc.cpu_pct:.1f}", align=Qt.AlignRight | Qt.AlignVCenter)
            if proc.cpu_pct >= 50:
                cpu_it.setForeground(QColor(_GREEN))
            self.proc_table.setItem(r, 5, cpu_it)
            self.proc_table.setItem(
                r, 6, _item(f"{proc.mem_pct:.1f}", align=Qt.AlignRight | Qt.AlignVCenter)
            )
            self.proc_table.setItem(r, 7, _item(proc.cmdline or proc.name))
