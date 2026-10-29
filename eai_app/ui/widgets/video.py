"""ui.widgets.video — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class RynnValueQtHudWindow(_eai_platform_bootstrap.QWidget):
    """Top-level Live HUD popup (Qt).

    EAI / ros-humble ships headless OpenCV (no HighGUI), so Isaac/RoboDojo
    cannot use ``cv2.namedWindow``. This window shows the same overlay RGB.
    """

    closed_by_user = _eai_platform_bootstrap.pyqtSignal()

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(
            parent,
            _eai_platform_bootstrap.Qt.Window | _eai_platform_bootstrap.Qt.WindowStaysOnTopHint,
        )
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("RynnValue Live HUD")
        self.setMinimumSize(480, 360)
        self.resize(640, 480)
        layout = _eai_platform_bootstrap.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = _eai_platform_bootstrap.QLabel("等待 Isaac cam_head.npy…")
        self._label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self._label.setMinimumSize(320, 240)
        self._label.setStyleSheet(
            f"background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: {_eai_configuration_policies.UI_TEXT_MUTED};"
        )
        layout.addWidget(self._label)
        self._user_closed = False

    def set_overlay_rgb(self, rgb: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]) -> None:
        if rgb is None or self._user_closed:
            return
        arr = _eai_platform_bootstrap.np.ascontiguousarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return
        h, w, _ch = arr.shape
        qimg = _eai_platform_bootstrap.QImage(
            arr.data, w, h, 3 * w, _eai_platform_bootstrap.QImage.Format_RGB888
        ).copy()
        pix = _eai_platform_bootstrap.QPixmap.fromImage(qimg)
        self._label.setPixmap(
            pix.scaled(
                self._label.size(),
                _eai_platform_bootstrap.Qt.KeepAspectRatio,
                _eai_platform_bootstrap.Qt.SmoothTransformation,
            )
        )

    def set_status_title(
        self,
        remaining_s: _eai_platform_bootstrap.Optional[float] = None,
        error: str = "",
    ) -> None:
        err = (error or "").strip()
        if err:
            self.setWindowTitle(f"RynnValue  ERR: {err[:64]}")
        elif remaining_s is not None:
            self.setWindowTitle(f"RynnValue  rem={float(remaining_s):.2f}s")
        else:
            self.setWindowTitle("RynnValue  waiting…")

    def closeEvent(self, event: _eai_platform_bootstrap.QCloseEvent) -> None:  # type: ignore[override]
        if not self._user_closed:
            self._user_closed = True
            self.closed_by_user.emit()
        super().closeEvent(event)


class VideoPlayerWidget(_eai_platform_bootstrap.QWidget):
    """内嵌 OpenCV 视频播放（暂停 / 重头 / 循环 / 可拖动进度条）。"""

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(parent)
        self._path = ""
        self._cap: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.cv2.VideoCapture] = None
        self._playing = False
        self._interval_ms = 33
        self._fps = 30.0
        self._frame_count = 0
        self._current_frame = 0
        self._scrubbing = False

        layout = _eai_platform_bootstrap.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._label = _eai_platform_bootstrap.QLabel("请选择视频…")
        self._label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self._label.setMinimumSize(480, 270)
        self._label.setStyleSheet("background-color: #111; color: #ccc;")
        layout.addWidget(self._label, 1)

        prog = _eai_platform_bootstrap.QHBoxLayout()
        prog.setSpacing(6)
        self._time_label = _eai_platform_bootstrap.QLabel("00:00.00 / 00:00.00")
        self._time_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        self._time_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        self._time_label.setMinimumWidth(130)
        prog.addWidget(self._time_label)
        self._slider = _eai_platform_bootstrap.QSlider(_eai_platform_bootstrap.Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)
        self._slider.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self._slider.setToolTip("拖动进度条跳转")
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        prog.addWidget(self._slider, 1)
        layout.addLayout(prog)

        ctrl = _eai_platform_bootstrap.QHBoxLayout()
        self._play_btn = _eai_platform_bootstrap.QPushButton("播放")
        self._play_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self._play_btn.clicked.connect(self.toggle_play)
        ctrl.addWidget(self._play_btn)
        self._restart_btn = _eai_platform_bootstrap.QPushButton("重头")
        self._restart_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self._restart_btn.clicked.connect(self.restart)
        ctrl.addWidget(self._restart_btn)
        self._info = _eai_platform_bootstrap.QLabel("")
        self._info.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        self._info.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        self._info.setWordWrap(True)
        ctrl.addWidget(self._info, 1)
        layout.addLayout(ctrl)

        self._timer = _eai_platform_bootstrap.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    def video_path(self) -> str:
        return self._path

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def is_playing(self) -> bool:
        return self._playing and self.is_open()

    @staticmethod
    def _format_time(frame: int, fps: float) -> str:
        if fps <= 1e-6:
            return f"{max(0, frame)}"
        secs = max(0.0, float(frame) / fps)
        minutes = int(secs // 60)
        rem = secs - minutes * 60
        return f"{minutes:02d}:{rem:05.2f}"

    def _update_time_label(self) -> None:
        cur = self._format_time(self._current_frame, self._fps)
        total_frames = max(0, self._frame_count - 1) if self._frame_count > 0 else 0
        tot = self._format_time(total_frames, self._fps)
        self._time_label.setText(f"{cur} / {tot}")

    def _set_slider_frame(self, frame: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(max(0, min(frame, self._slider.maximum())))
        self._slider.blockSignals(False)
        self._update_time_label()

    def _show_bgr_frame(self, frame) -> None:
        rgb = _eai_platform_bootstrap.cv2.cvtColor(frame, _eai_platform_bootstrap.cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = _eai_platform_bootstrap.QImage(rgb.data, w, h, ch * w, _eai_platform_bootstrap.QImage.Format_RGB888).copy()
        pix = _eai_platform_bootstrap.QPixmap.fromImage(qimg)
        self._label.setPixmap(
            pix.scaled(self._label.size(), _eai_platform_bootstrap.Qt.KeepAspectRatio, _eai_platform_bootstrap.Qt.SmoothTransformation)
        )

    def _seek_to_frame(self, frame: int, *, preview: bool = True) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        if self._frame_count > 0:
            frame = max(0, min(int(frame), self._frame_count - 1))
        else:
            frame = max(0, int(frame))
        self._cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_POS_FRAMES, frame)
        self._current_frame = frame
        self._set_slider_frame(frame)
        if not preview:
            return
        ok, img = self._cap.read()
        if not ok or img is None:
            return
        self._show_bgr_frame(img)
        # read 会前进一帧；暂停时回到目标帧，便于继续播放/再拖动
        if not self._playing:
            self._cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_POS_FRAMES, frame)
            self._current_frame = frame

    def load(self, video_path: str) -> bool:
        self.stop()
        self._path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(video_path or ""))
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_count = 0
        self._current_frame = 0
        self._slider.setEnabled(False)
        self._slider.setRange(0, 0)
        self._update_time_label()
        if not self._path or not _eai_platform_bootstrap.os.path.isfile(self._path):
            self._label.setText("视频不存在")
            self._info.setText(self._path or "")
            return False
        cap = _eai_platform_bootstrap.cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._label.setText(f"无法打开视频:\n{self._path}")
            self._info.setText(self._path)
            return False
        self._cap = cap
        fps = float(cap.get(_eai_platform_bootstrap.cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 1e-3 else 30.0
        self._interval_ms = max(15, int(1000.0 / self._fps))
        frame_count = int(cap.get(_eai_platform_bootstrap.cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_count = max(0, frame_count)
        max_idx = max(0, self._frame_count - 1) if self._frame_count > 0 else 0
        self._slider.setRange(0, max_idx)
        self._slider.setEnabled(self._frame_count > 0)
        self._info.setText(self._path)
        self._playing = False
        self._play_btn.setText("播放")
        self._seek_to_frame(0, preview=True)
        return True

    def play(self) -> None:
        if not self.is_open():
            return
        self._playing = True
        self._play_btn.setText("暂停")
        if not self._timer.isActive():
            self._timer.start(self._interval_ms)

    def pause(self) -> None:
        self._playing = False
        self._play_btn.setText("播放")
        self._timer.stop()

    def stop(self) -> None:
        self.pause()

    def toggle_play(self) -> None:
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def restart(self) -> None:
        self._seek_to_frame(0, preview=True)
        self.play()

    def close_video(self) -> None:
        self.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._slider.setEnabled(False)

    def _on_slider_pressed(self) -> None:
        self._scrubbing = True

    def _on_slider_released(self) -> None:
        self._scrubbing = False
        self._seek_to_frame(self._slider.value(), preview=True)

    def _on_slider_moved(self, value: int) -> None:
        if self._scrubbing:
            self._current_frame = int(value)
            self._update_time_label()
            self._seek_to_frame(value, preview=True)

    def _on_tick(self) -> None:
        if self._cap is None or not self._cap.isOpened():
            return
        if not self._playing or self._scrubbing:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            # 循环到开头
            self._cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame = 0
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return
        pos = int(self._cap.get(_eai_platform_bootstrap.cv2.CAP_PROP_POS_FRAMES) or 0)
        # POS_FRAMES 为下一帧索引
        self._current_frame = max(0, pos - 1)
        if not self._scrubbing:
            self._set_slider_frame(self._current_frame)
        self._show_bgr_frame(frame)


class SimEvalVideoDialog(_eai_platform_bootstrap.QDialog):
    """用 OpenCV 在窗口内播放评测 mp4（本机常无桌面播放器，xdg-open 会失败）。"""

    def __init__(self, video_path: str, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_eai_platform_bootstrap.os.path.basename(video_path))
        self.resize(960, 600)
        layout = _eai_platform_bootstrap.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._player = VideoPlayerWidget(self)
        layout.addWidget(self._player)
        if self._player.load(video_path):
            self._player.play()

    def closeEvent(self, event: _eai_platform_bootstrap.QCloseEvent) -> None:
        self._player.close_video()
        super().closeEvent(event)

