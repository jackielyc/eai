"""adapters.visualization.rerun — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_datasets_notes = module_ref('eai_app.ui.datasets.notes')

class RerunWebEmbedServer(_eai_platform_bootstrap.QObject):
    """后台 `rerun --serve-web`，供页内 QWebEngineView 连接（不弹原生窗口）。"""

    ready = _eai_platform_bootstrap.pyqtSignal(str)  # viewer http url
    failed = _eai_platform_bootstrap.pyqtSignal(str)
    stopped = _eai_platform_bootstrap.pyqtSignal()

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QObject] = None) -> None:
        super().__init__(parent)
        self._process: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QProcess] = None
        self._viewer_url = ""
        self._rrd_path = ""
        self._ready_emitted = False
        self._fallback_timer: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QTimer] = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() == _eai_platform_bootstrap.QProcess.Running

    def current_url(self) -> str:
        return self._viewer_url

    def current_rrd(self) -> str:
        return self._rrd_path

    def stop(self) -> None:
        if self._fallback_timer is not None:
            self._fallback_timer.stop()
            self._fallback_timer = None
        proc = self._process
        self._process = None
        self._viewer_url = ""
        self._ready_emitted = False
        if proc is not None:
            proc.readyReadStandardOutput.disconnect()
            proc.readyReadStandardError.disconnect()
            proc.finished.disconnect()
            proc.errorOccurred.disconnect()
            if proc.state() != _eai_platform_bootstrap.QProcess.NotRunning:
                proc.terminate()
                if not proc.waitForFinished(2500):
                    proc.kill()
                    proc.waitForFinished(1000)
            proc.deleteLater()
        self.stopped.emit()

    def start(self, rrd_path: str) -> None:
        path = _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(rrd_path or ""))
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            self.failed.emit(f"RRD 不存在: {path}")
            return
        argv = _eai_ui_datasets_notes.resolve_rerun_argv()
        if not argv:
            self.failed.emit(
                "未找到 Rerun。请安装 rerun-sdk，或设置 RERUN_BIN。"
            )
            return

        self.stop()
        web_port = _eai_ui_datasets_notes._pick_free_tcp_port()
        grpc_port = _eai_ui_datasets_notes._pick_free_tcp_port()
        while grpc_port == web_port:
            grpc_port = _eai_ui_datasets_notes._pick_free_tcp_port()

        proxy = f"rerun+http://127.0.0.1:{grpc_port}/proxy"
        self._viewer_url = (
            f"http://127.0.0.1:{web_port}/?url="
            + _eai_platform_bootstrap.urllib.parse.quote(proxy, safe="")
        )
        self._rrd_path = path
        self._ready_emitted = False

        cmd = list(argv) + [
            "--serve-web",
            "--bind",
            "127.0.0.1",
            "--web-viewer-port",
            str(web_port),
            "--port",
            str(grpc_port),
            path,
        ]
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.readyReadStandardError.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._process = proc
        proc.start(cmd[0], cmd[1:])
        if not proc.waitForStarted(8000):
            detail = bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
            self.stop()
            self.failed.emit(f"启动 serve-web 失败: {detail[:300] or cmd[0]}")
            return

        # 大文件加载可能较慢：日志就绪 或 超时兜底加载 URL
        self._fallback_timer = _eai_platform_bootstrap.QTimer(self)
        self._fallback_timer.setSingleShot(True)
        self._fallback_timer.timeout.connect(self._emit_ready_once)
        self._fallback_timer.start(2500)

    def _emit_ready_once(self) -> None:
        if self._ready_emitted or not self.is_running() or not self._viewer_url:
            return
        self._ready_emitted = True
        if self._fallback_timer is not None:
            self._fallback_timer.stop()
            self._fallback_timer = None
        self.ready.emit(self._viewer_url)

    def _on_output(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        if "Hosting a web-viewer" in text or "web-viewer at" in text:
            self._emit_ready_once()

    def _on_finished(self, exit_code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
        if not self._ready_emitted and exit_code != 0:
            self.failed.emit(f"Rerun serve-web 退出 (code={exit_code})")
        self._process = None
        self._viewer_url = ""
        self.stopped.emit()

    def _on_error(self, error: _eai_platform_bootstrap.QProcess.ProcessError) -> None:
        if error == _eai_platform_bootstrap.QProcess.FailedToStart:
            self.failed.emit("无法启动 Rerun 进程（FailedToStart）")

