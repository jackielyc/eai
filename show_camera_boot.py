#!/usr/bin/env python3
"""Fast splash launcher for show_camera_topics.py.

Shows a minimal Qt window within ~100ms, then loads the heavy viewer module
(cv2 / rclpy / 3.5万行 UI) while the splash stays visible.
"""
from __future__ import annotations

import os
import sys
import time


def _prepare_qt_env() -> None:
    os.environ.pop("QT_PLUGIN_PATH", None)
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        val = os.environ.get(key, "")
        if "/cv2/" in val.replace("\\", "/"):
            os.environ.pop(key, None)

    os.environ.setdefault("QT_X11_NO_MITSHM", "1")
    # 勿默认 QT_XCB_GL_INTEGRATION=none：本机 DISPLAY 有 GLX（含 llvmpipe）时，
    # 强制 none + AA_ShareOpenGLContexts 会报 “neither GLX nor EGL are enabled”
    # 并在创建 QWebEngineView 时偶发 SIP segfault。无 GLX 的远程 X 请显式 export。
    if not os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS"):
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            "--disable-gpu --disable-gpu-compositing --disable-webgl "
            "--disable-dev-shm-usage --num-raster-threads=2"
        )
    # IME：与 show_camera_topics.configure_qt_ime_for_chinese 对齐（须在 QApplication 前）
    # 不可硬编码 zh_CN.UTF-8：本机未安装时 LC_CTYPE 失效，剪贴板中文会乱码。
    def _pick_utf8_locale() -> str:
        import locale as _locale

        for name in (
            "zh_CN.UTF-8",
            "zh_CN.utf8",
            "C.UTF-8",
            "C.utf8",
            "en_US.UTF-8",
            "en_US.utf8",
        ):
            try:
                _locale.setlocale(_locale.LC_ALL, name)
                return name
            except _locale.Error:
                continue
        return "C.UTF-8"

    _loc = _pick_utf8_locale()
    os.environ.setdefault("LANG", _loc)
    os.environ.setdefault("LC_ALL", _loc)
    os.environ.setdefault("QT_IM_MODULE", "fcitx")
    os.environ.setdefault("XMODIFIERS", "@im=fcitx")
    os.environ.setdefault("GTK_IM_MODULE", "fcitx")

    try:
        import PyQt5

        root = os.path.dirname(os.path.abspath(PyQt5.__file__))
        qt_lib = os.path.join(root, "Qt5", "lib")
        if os.path.isdir(qt_lib):
            cur = os.environ.get("LD_LIBRARY_PATH", "")
            parts = [p for p in cur.split(":") if p and p != qt_lib]
            os.environ["LD_LIBRARY_PATH"] = ":".join([qt_lib] + parts)
        plug = os.path.join(root, "Qt5", "plugins")
        if os.path.isdir(plug):
            from PyQt5.QtCore import QCoreApplication

            QCoreApplication.addLibraryPath(plug)
    except Exception:
        pass


def _make_splash(app):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

    splash = QWidget()
    splash.setObjectName("bootSplash")
    splash.setWindowTitle("EAI · Camera Topic Viewer")
    splash.setFixedSize(440, 200)
    splash.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
    splash.setStyleSheet(
        "#bootSplash {"
        "  background-color: #1e1f22;"
        "  border: 1px solid #3d4048;"
        "}"
        "#bootSplash QLabel#bootBrand {"
        "  color: #7ec8ff;"
        "  letter-spacing: 2px;"
        "}"
        "#bootSplash QLabel#bootTitle {"
        "  color: #ececec;"
        "}"
        "#bootSplash QLabel#bootStatus {"
        "  color: #b0b0b0;"
        "}"
        "#bootSplash QProgressBar {"
        "  border: 1px solid #3d4048;"
        "  border-radius: 3px;"
        "  background: #1a1b1e;"
        "  max-height: 6px;"
        "  text-align: center;"
        "}"
        "#bootSplash QProgressBar::chunk {"
        "  background-color: #4da3ff;"
        "  border-radius: 2px;"
        "}"
    )
    layout = QVBoxLayout(splash)
    layout.setContentsMargins(28, 28, 28, 24)
    layout.setSpacing(8)
    brand = QLabel("EAI")
    brand.setObjectName("bootBrand")
    brand.setFont(QFont("", 11, QFont.Bold))
    brand.setAlignment(Qt.AlignCenter)
    title = QLabel("Camera Topic Viewer")
    title.setObjectName("bootTitle")
    title.setFont(QFont("", 15, QFont.Bold))
    title.setAlignment(Qt.AlignCenter)
    status = QLabel("正在加载…")
    status.setObjectName("bootStatus")
    status.setAlignment(Qt.AlignCenter)
    bar = QProgressBar()
    bar.setRange(0, 0)  # indeterminate
    bar.setTextVisible(False)
    layout.addWidget(brand)
    layout.addWidget(title)
    layout.addSpacing(4)
    layout.addWidget(bar)
    layout.addWidget(status)
    splash._boot_status = status  # type: ignore[attr-defined]
    splash._boot_t0 = time.perf_counter()  # type: ignore[attr-defined]
    splash.show()
    splash.raise_()
    app.processEvents()
    return splash


def _set_status(splash, app, text: str) -> None:
    try:
        elapsed = time.perf_counter() - getattr(splash, "_boot_t0", time.perf_counter())
        splash._boot_status.setText(f"{text}  ({elapsed:.1f}s)")  # type: ignore[attr-defined]
        app.processEvents()
    except Exception:
        pass


def main() -> int:
    _prepare_qt_env()

    from PyQt5.QtCore import Qt, QCoreApplication
    from PyQt5.QtWidgets import QApplication

    # WebEngine 必须在 QApplication 之前 import（否则部分环境初始化失败）
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView  # noqa: F401
    except Exception:
        pass

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    splash = _make_splash(app)

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    _set_status(splash, app, "加载界面模块…")
    import show_camera_topics as sct  # noqa: WPS433 — intentional deferred heavy import

    _set_status(splash, app, "初始化…")
    return sct.main(existing_app=app, splash=splash)


if __name__ == "__main__":
    sys.exit(main())
