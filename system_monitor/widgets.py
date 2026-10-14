"""htop / nvtop 风格自定义绘制控件。"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

_BG = QColor("#1a1b1e")
_PANEL = QColor("#252628")
_BORDER = QColor("#3d4048")
_TEXT = QColor("#ececec")
_MUTED = QColor("#b0b0b0")
_GREEN = QColor("#50fa7b")
_ORANGE = QColor("#ffb86c")
_RED = QColor("#ff8888")
_BLUE = QColor("#7ec8ff")
_CYAN = QColor("#8be9fd")
_YELLOW = QColor("#f1fa8c")
_HTOP_USED = QColor("#50fa7b")  # used
_HTOP_BUFFERS = QColor("#f1fa8c")  # buffers
_HTOP_CACHE = QColor("#bd93f9")  # cache


def _util_color(pct: float) -> QColor:
    p = max(0.0, min(100.0, float(pct)))
    if p < 50.0:
        t = p / 50.0
        return QColor(
            int(_GREEN.red() + (_ORANGE.red() - _GREEN.red()) * t),
            int(_GREEN.green() + (_ORANGE.green() - _GREEN.green()) * t),
            int(_GREEN.blue() + (_ORANGE.blue() - _GREEN.blue()) * t),
        )
    t = (p - 50.0) / 50.0
    return QColor(
        int(_ORANGE.red() + (_RED.red() - _ORANGE.red()) * t),
        int(_ORANGE.green() + (_RED.green() - _ORANGE.green()) * t),
        int(_ORANGE.blue() + (_RED.blue() - _ORANGE.blue()) * t),
    )


class HtopCpuMeters(QWidget):
    """htop 风格：每核一条水平进度条，多列排布。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cores: List[float] = []
        self._avg = 0.0
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_cores(self, values: Sequence[float], avg: Optional[float] = None) -> None:
        self._cores = [max(0.0, min(100.0, float(v))) for v in values]
        if avg is None:
            self._avg = (
                sum(self._cores) / len(self._cores) if self._cores else 0.0
            )
        else:
            self._avg = max(0.0, min(100.0, float(avg)))
        # Grow height with core count (but clamp).
        n = len(self._cores)
        cols = max(1, min(8, (n + 47) // 48)) if n else 1
        rows = (n + cols - 1) // cols if n else 1
        self.setMinimumHeight(max(90, min(360, 28 + rows * 14)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, _PANEL)
        p.setPen(QPen(_BORDER, 1))
        p.drawRect(0, 0, w - 1, h - 1)

        # Avg meter (htop top line)
        pad = 6
        y = 4
        p.setPen(_MUTED)
        p.setFont(QFont("Monospace", 9, QFont.Bold))
        label = "Avg"
        p.drawText(QRectF(pad, y, 28, 14), Qt.AlignLeft | Qt.AlignVCenter, label)
        self._draw_meter(p, QRectF(pad + 30, y + 1, w - pad * 2 - 78, 12), self._avg)
        p.setPen(_TEXT)
        p.drawText(
            QRectF(w - pad - 46, y, 46, 14),
            Qt.AlignRight | Qt.AlignVCenter,
            f"{self._avg:5.1f}%",
        )
        y += 18

        n = len(self._cores)
        if n == 0:
            p.setPen(_MUTED)
            p.drawText(QRectF(0, y, w, h - y), Qt.AlignCenter, "等待 CPU 采样…")
            return

        # Multi-column per-core meters
        avail_h = h - y - 4
        # Prefer ~12–14px row; choose cols so bars fit.
        row_h = 13
        max_rows = max(1, int(avail_h // row_h))
        cols = max(1, (n + max_rows - 1) // max_rows)
        cols = min(cols, 12)
        rows = (n + cols - 1) // cols
        col_w = (w - 2 * pad) / cols

        p.setFont(QFont("Monospace", 8))
        for i, val in enumerate(self._cores):
            r = i % rows
            c = i // rows
            x0 = pad + c * col_w
            yy = y + r * row_h
            idx_w = 28 if n < 1000 else 34
            p.setPen(_MUTED)
            p.drawText(
                QRectF(x0, yy, idx_w, row_h - 1),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{i}",
            )
            meter = QRectF(x0 + idx_w + 3, yy + 1.5, col_w - idx_w - 42, row_h - 4)
            self._draw_meter(p, meter, val)
            p.setPen(_TEXT)
            p.drawText(
                QRectF(x0 + col_w - 38, yy, 36, row_h - 1),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{val:4.0f}%",
            )

    @staticmethod
    def _draw_meter(p: QPainter, rect: QRectF, pct: float) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(_BG)
        p.drawRect(rect)
        fill_w = rect.width() * (max(0.0, min(100.0, pct)) / 100.0)
        if fill_w > 0:
            # htop-like: solid green→yellow→red blocks feel like |||||
            fill = QRectF(rect.x(), rect.y(), fill_w, rect.height())
            p.setBrush(_util_color(pct))
            p.drawRect(fill)
        p.setPen(QPen(_BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)


class HtopMemBar(QWidget):
    """htop 风格 Mem/Swap 分段条：used | buffers | cache。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = "Mem"
        self._total = 1.0
        self._used = 0.0
        self._buffers = 0.0
        self._cached = 0.0
        self._detail = ""
        self.setMinimumHeight(28)
        self.setMaximumHeight(34)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_mem(
        self,
        *,
        title: str,
        total: float,
        used: float,
        buffers: float = 0.0,
        cached: float = 0.0,
        detail: str = "",
    ) -> None:
        self._title = title
        self._total = max(1.0, float(total))
        self._used = max(0.0, float(used))
        self._buffers = max(0.0, float(buffers))
        self._cached = max(0.0, float(cached))
        self._detail = detail
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        pad = 4
        p.setPen(_MUTED)
        p.setFont(QFont("Monospace", 9, QFont.Bold))
        p.drawText(QRectF(pad, 2, 40, h - 4), Qt.AlignLeft | Qt.AlignVCenter, self._title)

        bar = QRectF(pad + 42, 6, w - pad * 2 - 42 - 8, h - 12)
        p.setPen(Qt.NoPen)
        p.setBrush(_BG)
        p.drawRect(bar)

        x = bar.x()
        segs = (
            (self._used, _HTOP_USED),
            (self._buffers, _HTOP_BUFFERS),
            (self._cached, _HTOP_CACHE),
        )
        for amount, color in segs:
            frac = min(1.0, amount / self._total)
            sw = bar.width() * frac
            if sw > 0:
                p.setBrush(color)
                p.drawRect(QRectF(x, bar.y(), sw, bar.height()))
                x += sw

        p.setPen(QPen(_BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar)

        if self._detail:
            p.setPen(_TEXT)
            p.setFont(QFont("Monospace", 8))
            p.drawText(bar.adjusted(4, 0, -4, 0), Qt.AlignRight | Qt.AlignVCenter, self._detail)


class NvtopGpuPanel(QWidget):
    """单卡 nvtop 风格面板：GPU/MEM 条 + 时钟/温度/功耗 + 历史图。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = "GPU"
        self._gpu_pct = 0.0
        self._mem_pct = 0.0
        self._mem_detail = ""
        self._meta = ""
        self._gpu_hist: Deque[float] = deque(maxlen=90)
        self._mem_hist: Deque[float] = deque(maxlen=90)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_stats(
        self,
        *,
        title: str,
        gpu_pct: float,
        mem_pct: float,
        mem_detail: str,
        meta: str,
    ) -> None:
        self._title = title
        self._gpu_pct = max(0.0, min(100.0, float(gpu_pct)))
        self._mem_pct = max(0.0, min(100.0, float(mem_pct)))
        self._mem_detail = mem_detail
        self._meta = meta
        self._gpu_hist.append(self._gpu_pct)
        self._mem_hist.append(self._mem_pct)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, _PANEL)
        p.setPen(QPen(_BORDER, 1))
        p.drawRect(0, 0, w - 1, h - 1)

        pad = 8
        y = 4
        p.setPen(_CYAN)
        p.setFont(QFont("Monospace", 9, QFont.Bold))
        p.drawText(QRectF(pad, y, w - 2 * pad, 16), Qt.AlignLeft | Qt.AlignVCenter, self._title)
        y += 18

        # GPU / MEM meters
        p.setFont(QFont("Monospace", 8))
        for label, pct, color, detail in (
            ("GPU", self._gpu_pct, _GREEN, f"{self._gpu_pct:5.1f}%"),
            ("MEM", self._mem_pct, _ORANGE, self._mem_detail or f"{self._mem_pct:5.1f}%"),
        ):
            p.setPen(_MUTED)
            p.drawText(QRectF(pad, y, 28, 14), Qt.AlignLeft | Qt.AlignVCenter, label)
            bar = QRectF(pad + 30, y + 2, w - pad * 2 - 30 - 110, 10)
            p.setPen(Qt.NoPen)
            p.setBrush(_BG)
            p.drawRect(bar)
            fw = bar.width() * (pct / 100.0)
            if fw > 0:
                p.setBrush(color)
                p.drawRect(QRectF(bar.x(), bar.y(), fw, bar.height()))
            p.setPen(QPen(_BORDER, 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(bar)
            p.setPen(_TEXT)
            p.drawText(
                QRectF(w - pad - 108, y, 108, 14),
                Qt.AlignRight | Qt.AlignVCenter,
                detail,
            )
            y += 16

        p.setPen(_MUTED)
        p.setFont(QFont("Monospace", 8))
        p.drawText(QRectF(pad, y, w - 2 * pad, 14), Qt.AlignLeft | Qt.AlignVCenter, self._meta)
        y += 16

        # Dual history plot (nvtop style)
        plot = QRectF(pad, y, w - 2 * pad, max(40.0, h - y - pad))
        p.setPen(Qt.NoPen)
        p.setBrush(_BG)
        p.drawRect(plot)
        p.setPen(QPen(_BORDER, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(plot)

        def _draw_hist(buf: Deque[float], color: QColor) -> None:
            if len(buf) < 2:
                return
            path = QPainterPath()
            nmax = buf.maxlen or len(buf)
            for i, val in enumerate(buf):
                x = plot.left() + plot.width() * i / max(1, nmax - 1)
                yy = plot.bottom() - plot.height() * (val / 100.0)
                if i == 0:
                    path.moveTo(x, yy)
                else:
                    path.lineTo(x, yy)
            p.setPen(QPen(color, 1.5))
            p.drawPath(path)

        _draw_hist(self._gpu_hist, _GREEN)
        _draw_hist(self._mem_hist, _ORANGE)

        # legend
        p.setFont(QFont("Monospace", 7))
        p.setPen(_GREEN)
        p.drawText(QRectF(plot.left() + 4, plot.top() + 2, 40, 12), Qt.AlignLeft, "GPU")
        p.setPen(_ORANGE)
        p.drawText(QRectF(plot.left() + 36, plot.top() + 2, 40, 12), Qt.AlignLeft, "MEM")


class HistoryChart(QWidget):
    """多序列历史折线（0–100%）。"""

    def __init__(
        self,
        title: str = "",
        parent: Optional[QWidget] = None,
        *,
        max_points: int = 120,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._max_points = max(20, int(max_points))
        self._series: List[Tuple[str, QColor, Deque[float]]] = []
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def ensure_series(self, names_colors: Sequence[Tuple[str, QColor]]) -> None:
        existing = {s[0]: s for s in self._series}
        rebuilt: List[Tuple[str, QColor, Deque[float]]] = []
        for name, color in names_colors:
            if name in existing:
                _, _, buf = existing[name]
                rebuilt.append((name, color, buf))
            else:
                rebuilt.append((name, color, deque(maxlen=self._max_points)))
        self._series = rebuilt

    def push(self, values: Sequence[float]) -> None:
        for i, series in enumerate(self._series):
            val = float(values[i]) if i < len(values) else 0.0
            series[2].append(max(0.0, min(100.0, val)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, _PANEL)
        p.setPen(QPen(_BORDER, 1))
        p.drawRect(0, 0, w - 1, h - 1)

        pad_l, pad_r, pad_t, pad_b = 36, 10, 20, 14
        plot = QRectF(pad_l, pad_t, w - pad_l - pad_r, h - pad_t - pad_b)

        p.setPen(_MUTED)
        p.setFont(QFont("Sans Serif", 9, QFont.Bold))
        p.drawText(QRectF(8, 1, w - 16, 16), Qt.AlignLeft | Qt.AlignVCenter, self._title)

        lx = w - 10
        p.setFont(QFont("Sans Serif", 8))
        for name, color, _buf in reversed(self._series):
            tw = p.fontMetrics().horizontalAdvance(name) + 14
            lx -= tw
            p.setPen(color)
            p.drawText(QRectF(lx + 10, 1, tw - 10, 16), Qt.AlignLeft | Qt.AlignVCenter, name)
            p.fillRect(int(lx), 6, 8, 6, color)

        p.setPen(QPen(QColor(61, 64, 72, 120), 1, Qt.DotLine))
        for frac in (0.0, 0.5, 1.0):
            y = plot.bottom() - plot.height() * frac
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(_MUTED)
            p.setFont(QFont("Monospace", 7))
            p.drawText(
                QRectF(0, y - 7, pad_l - 4, 14),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{int(frac * 100)}",
            )
            p.setPen(QPen(QColor(61, 64, 72, 120), 1, Qt.DotLine))

        for _name, color, buf in self._series:
            if len(buf) < 2:
                continue
            path = QPainterPath()
            for i, val in enumerate(buf):
                x = plot.left() + (plot.width() * i / max(1, self._max_points - 1))
                y = plot.bottom() - plot.height() * (val / 100.0)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(color, 1.8))
            p.drawPath(path)
