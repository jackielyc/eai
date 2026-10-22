"""FLORA 结果可视化：时间轴 + 片段列表 + 视频帧预览。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

_TEXT = "#ececec"
_MUTED = "#b0b0b0"
_BG = "#1a1b1e"
_BORDER = "#3d4048"
_ACCENT = "#6cb6ff"
_PLAYHEAD = "#ffdd57"

# annotate segment palette
_SEG_COLORS = [
    "#4fc3f7",
    "#81c784",
    "#ffb74d",
    "#ba68c8",
    "#4db6ac",
    "#f06292",
    "#9575cd",
    "#aed581",
]

_PROCESS_COLORS = {
    "failure": "#e57373",
    "suboptimal": "#ffb74d",
    "normal": "#81c784",
}


@dataclass
class VizSegment:
    start_s: float
    end_s: float
    title: str
    detail: str = ""
    color: str = _SEG_COLORS[0]
    score: Optional[float] = None
    label: str = ""


@dataclass
class VizModel:
    kind: str  # annotate | outcome | process | unknown
    path: str
    duration_s: float
    summary: str
    segments: List[VizSegment] = field(default_factory=list)
    outcome: Optional[dict] = None


def _fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    m = int(t // 60)
    s = t - 60 * m
    return f"{m}:{s:05.2f}" if m else f"{s:.2f}s"


def load_viz_model(path: Path) -> VizModel:
    path = Path(path).expanduser()
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return VizModel("unknown", str(path), 0.0, "空 JSONL", [])
        data = json.loads(lines[0])
        return _from_jsonl_record(data, str(path))
    data = json.loads(raw)
    return _from_annotation(data, str(path))


def _from_annotation(data: dict, path: str) -> VizModel:
    segs_raw = data.get("segments") or []
    duration = float(data.get("duration_s") or 0.0)
    if not duration and segs_raw:
        duration = max(float(s.get("end_s") or 0) for s in segs_raw)
    segments: List[VizSegment] = []
    for i, s in enumerate(segs_raw):
        start = float(s.get("start_s") or 0)
        end = float(s.get("end_s") or start)
        caption = str(s.get("caption") or "").strip() or "(no caption)"
        status = str(s.get("status") or "")
        prov = str(s.get("segmentation_provenance") or "")
        detail_bits = [x for x in (status, prov) if x]
        segments.append(
            VizSegment(
                start_s=start,
                end_s=end,
                title=caption,
                detail=" · ".join(detail_bits),
                color=_SEG_COLORS[i % len(_SEG_COLORS)],
            )
        )
    profile = data.get("profile") or ""
    summary = (
        f"Annotate · {len(segments)} segments · {_fmt_time(duration)}"
        + (f" · profile={profile}" if profile else "")
    )
    return VizModel("annotate", path, duration, summary, segments)


def _from_jsonl_record(data: dict, path: str) -> VizModel:
    if "probabilities" in data or ("score" in data and "segments" not in data):
        return _from_outcome(data, path)
    if "segments" in data and ("worst_score" in data or "mean_score" in data or "fraction" in data):
        return _from_process(data, path)
    # rubric / generic
    segs = data.get("segments") or []
    if segs:
        return _from_process(data, path)
    duration = float(data.get("duration_s") or 0)
    summary = f"Reward · score={data.get('score')} · {_fmt_time(duration)}"
    return VizModel("outcome", path, duration, summary, [], outcome=data)


def _from_outcome(data: dict, path: str) -> VizModel:
    duration = float(data.get("duration_s") or 0)
    score = data.get("score")
    expected = data.get("expected_score")
    progress = data.get("progress")
    task = str(data.get("task") or "")
    bits = [f"Outcome score={score}"]
    if expected is not None:
        bits.append(f"E[score]={expected}")
    if progress is not None:
        bits.append(f"progress={progress}")
    if task:
        bits.append(f"task={task[:60]}")
    summary = " · ".join(bits) + f" · {_fmt_time(duration)}"
    # single full-span bar colored by score
    color = _PROCESS_COLORS["normal"]
    try:
        sc = int(score)
        if sc <= 2:
            color = _PROCESS_COLORS["failure"]
        elif sc == 3:
            color = _PROCESS_COLORS["suboptimal"]
    except (TypeError, ValueError):
        pass
    segments = [
        VizSegment(
            start_s=0.0,
            end_s=duration or 1.0,
            title=f"Episode score {score}",
            detail=str(data.get("output") or data.get("rationale") or ""),
            color=color,
            score=float(score) if score is not None else None,
            label="outcome",
        )
    ]
    return VizModel("outcome", path, duration, summary, segments, outcome=data)


def _from_process(data: dict, path: str) -> VizModel:
    segs_raw = data.get("segments") or []
    duration = float(data.get("duration_s") or 0)
    if not duration and segs_raw:
        duration = max(float(s.get("end_s") or 0) for s in segs_raw)
    segments: List[VizSegment] = []
    for s in segs_raw:
        start = float(s.get("start_s") or 0)
        end = float(s.get("end_s") or start)
        label = str(s.get("label") or "normal")
        score = s.get("score")
        rationale = str(s.get("rationale") or "").strip()
        title = f"[{label}] score={score}"
        if rationale:
            title = f"{title} — {rationale}"
        segments.append(
            VizSegment(
                start_s=start,
                end_s=end,
                title=title,
                detail=rationale,
                color=_PROCESS_COLORS.get(label, _SEG_COLORS[0]),
                score=float(score) if score is not None else None,
                label=label,
            )
        )
    frac = data.get("fraction") or {}
    summary = (
        f"Process · mean={data.get('mean_score')} worst={data.get('worst_score')} "
        f"· fail={frac.get('failure')} sub={frac.get('suboptimal')} ok={frac.get('normal')} "
        f"· {len(segments)} segs · {_fmt_time(duration)}"
    )
    return VizModel("process", path, duration, summary, segments, outcome=data)


class TimelineBar(QWidget):
    """Clickable segment timeline with playhead."""

    seekRequested = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setMaximumHeight(48)
        self._segments: List[VizSegment] = []
        self._duration = 1.0
        self._t = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("点击跳转到对应时间")

    def set_model(self, duration_s: float, segments: List[VizSegment]) -> None:
        self._duration = max(float(duration_s), 1e-3)
        self._segments = list(segments)
        self.update()

    def set_time(self, t: float) -> None:
        self._t = max(0.0, min(float(t), self._duration))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 8, -4, -8)
        p.fillRect(rect, QColor("#2a2c33"))
        p.setPen(QPen(QColor(_BORDER)))
        p.drawRect(rect)
        dur = self._duration
        for seg in self._segments:
            x0 = rect.left() + int(rect.width() * (seg.start_s / dur))
            x1 = rect.left() + int(rect.width() * (seg.end_s / dur))
            if x1 <= x0:
                x1 = x0 + 2
            p.fillRect(x0, rect.top(), max(2, x1 - x0), rect.height(), QColor(seg.color))
        # playhead
        x = rect.left() + int(rect.width() * (self._t / dur))
        p.setPen(QPen(QColor(_PLAYHEAD), 2))
        p.drawLine(x, rect.top() - 4, x, rect.bottom() + 4)
        p.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        rect = self.rect().adjusted(4, 8, -4, -8)
        if rect.width() <= 0:
            return
        x = max(rect.left(), min(event.x(), rect.right()))
        t = self._duration * (x - rect.left()) / rect.width()
        self.seekRequested.emit(t)


class FloraResultPanel(QWidget):
    """Embedded result viewer for annotate / outcome / process outputs."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("floraResultPanel")
        self.setStyleSheet(
            f"#floraResultPanel {{ background-color: {_BG}; }}"
            f"#floraResultPanel QLabel {{ color: {_TEXT}; background: transparent; }}"
            f"#floraResultPanel QListWidget {{ color: {_TEXT}; background-color: {_BG}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; }}"
        )
        self._model: Optional[VizModel] = None
        self._video: Optional[Path] = None
        self._cap = None
        self._fps = 30.0
        self._nframes = 0
        self._playing = False
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.summary_label = QLabel("尚无结果。跑完 annotate / outcome / process 后会显示在这里。")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {_MUTED};")
        root.addWidget(self.summary_label)

        body = QHBoxLayout()
        body.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(4)
        self.frame_label = QLabel("无视频预览")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(320, 180)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_label.setStyleSheet(
            f"QLabel {{ background-color: #111; border: 1px solid {_BORDER}; "
            f"border-radius: 4px; color: {_MUTED}; }}"
        )
        left.addWidget(self.frame_label, 1)

        ctrl = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(36)
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.play_btn)
        self.time_label = QLabel("0.00s")
        self.time_label.setFont(QFont("Monospace", 9))
        self.time_label.setMinimumWidth(70)
        ctrl.addWidget(self.time_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self.slider, 1)
        left.addLayout(ctrl)

        self.timeline = TimelineBar()
        self.timeline.seekRequested.connect(self._seek)
        left.addWidget(self.timeline)

        self.outcome_label = QLabel("")
        self.outcome_label.setWordWrap(True)
        self.outcome_label.setStyleSheet(f"color: {_ACCENT};")
        self.outcome_label.hide()
        left.addWidget(self.outcome_label)

        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(2)
        right.addWidget(QLabel("片段"))
        self.seg_list = QListWidget()
        self.seg_list.currentRowChanged.connect(self._on_seg_selected)
        right.addWidget(self.seg_list, 1)
        body.addLayout(right, 2)

        root.addLayout(body, 1)

        self._t = 0.0
        self._slider_block = False

    def clear(self) -> None:
        self._stop_play()
        self._close_cap()
        self._model = None
        self._video = None
        self.summary_label.setText("尚无结果。")
        self.seg_list.clear()
        self.timeline.set_model(1.0, [])
        self.outcome_label.hide()
        self.frame_label.setText("无视频预览")
        self.frame_label.setPixmap(QPixmap())

    def load(self, result_path: Path, video_path: Optional[Path] = None) -> None:
        self._stop_play()
        self._close_cap()
        model = load_viz_model(Path(result_path))
        self._model = model
        self.summary_label.setText(f"{model.summary}\n{model.path}")
        self.seg_list.clear()
        for i, seg in enumerate(model.segments):
            text = f"{i + 1:02d}  [{_fmt_time(seg.start_s)}–{_fmt_time(seg.end_s)}]  {seg.title}"
            item = QListWidgetItem(text)
            item.setForeground(QColor(seg.color))
            item.setData(Qt.UserRole, i)
            tip = seg.detail or seg.title
            item.setToolTip(tip)
            self.seg_list.addItem(item)
        self.timeline.set_model(model.duration_s or 1.0, model.segments)
        self._show_outcome(model)

        vid = Path(video_path).expanduser() if video_path else None
        if vid and vid.is_file():
            self._open_video(vid)
        else:
            self._video = None
            self.frame_label.setText("未绑定视频（在上方选择输入视频后可预览帧）")

        self._seek(0.0)

    def _show_outcome(self, model: VizModel) -> None:
        if model.kind != "outcome" or not model.outcome:
            self.outcome_label.hide()
            return
        o = model.outcome
        probs = o.get("probabilities") or {}
        lines = [f"Score {o.get('score')}   E={o.get('expected_score')}   progress={o.get('progress')}"]
        if probs:
            bar = "  ".join(f"{k}:{float(v):.0%}" for k, v in sorted(probs.items(), key=lambda x: int(x[0])))
            lines.append(bar)
        if o.get("rationale"):
            lines.append(str(o["rationale"]))
        elif o.get("output"):
            lines.append(str(o["output"]))
        self.outcome_label.setText("\n".join(lines))
        self.outcome_label.show()

    def _open_video(self, path: Path) -> None:
        try:
            import cv2
        except ImportError:
            self.frame_label.setText("需要 opencv (cv2) 才能预览视频帧")
            return
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.frame_label.setText(f"无法打开视频: {path}")
            return
        self._cap = cap
        self._video = path
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = fps
        self._nframes = n
        interval = max(16, int(1000 / fps))
        self._timer.setInterval(interval)
        if self._model and self._model.duration_s <= 0 and n > 0:
            self._model.duration_s = n / fps
            self.timeline.set_model(self._model.duration_s, self._model.segments)

    def _close_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._cap = None

    def _duration(self) -> float:
        if self._model and self._model.duration_s > 0:
            return self._model.duration_s
        if self._nframes > 0 and self._fps > 0:
            return self._nframes / self._fps
        return 1.0

    def _seek(self, t: float) -> None:
        dur = self._duration()
        self._t = max(0.0, min(float(t), dur))
        self.time_label.setText(_fmt_time(self._t))
        self.timeline.set_time(self._t)
        self._slider_block = True
        self.slider.setValue(int(1000 * self._t / dur))
        self._slider_block = False
        self._show_frame(self._t)
        self._sync_list_to_time()

    def _on_slider(self, value: int) -> None:
        if self._slider_block:
            return
        self._seek(self._duration() * (value / 1000.0))

    def _on_seg_selected(self, row: int) -> None:
        if not self._model or row < 0 or row >= len(self._model.segments):
            return
        seg = self._model.segments[row]
        self._seek(seg.start_s)

    def _sync_list_to_time(self) -> None:
        if not self._model:
            return
        t = self._t
        best = -1
        for i, seg in enumerate(self._model.segments):
            if seg.start_s <= t < seg.end_s or (i == len(self._model.segments) - 1 and abs(t - seg.end_s) < 1e-3):
                best = i
                break
        if best < 0:
            return
        if self.seg_list.currentRow() != best:
            self.seg_list.blockSignals(True)
            self.seg_list.setCurrentRow(best)
            self.seg_list.blockSignals(False)

    def _show_frame(self, t: float) -> None:
        if self._cap is None:
            return
        try:
            import cv2
        except ImportError:
            return
        frame_idx = int(round(t * self._fps))
        if self._nframes > 0:
            frame_idx = max(0, min(frame_idx, self._nframes - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(img)
        self.frame_label.setPixmap(
            pix.scaled(self.frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._cap is not None:
            self._show_frame(self._t)

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
        else:
            self._playing = True
            self.play_btn.setText("⏸")
            self._timer.start()

    def _stop_play(self) -> None:
        self._playing = False
        self.play_btn.setText("▶")
        self._timer.stop()

    def _on_tick(self) -> None:
        step = self._timer.interval() / 1000.0
        nxt = self._t + step
        if nxt >= self._duration():
            self._seek(self._duration())
            self._stop_play()
            return
        self._seek(nxt)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_play()
        self._close_cap()
        super().closeEvent(event)
