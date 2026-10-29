"""platform.qt environment — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def configure_qt_webengine_for_remote_display() -> None:
    """降低 WebEngine 对 GPU/GLX 的依赖，避免启动刷屏。

    须在创建 QApplication 之前调用。已有 QTWEBENGINE_CHROMIUM_FLAGS 时不覆盖。

    注意：不要默认设置 QT_XCB_GL_INTEGRATION=none。该值会禁用 GLX/EGL，
    与 AA_ShareOpenGLContexts 组合时常见
    “Cannot create platform OpenGL context” 并在创建 QWebEngineView
    时触发 SIP segfault。无 GLX 的远程桌面请自行 export QT_XCB_GL_INTEGRATION=none。
    """
    if _eai_platform_bootstrap.os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip():
        return
    # Gallery 等含 H.264 的页面仍建议用「浏览器打开」（Qt5 WebEngine 常无专有编解码）。
    _eai_platform_bootstrap.os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(
        [
            "--disable-gpu",
            "--disable-gpu-compositing",
            "--disable-webgl",
            "--disable-dev-shm-usage",
            "--num-raster-threads=2",
        ]
    )


def _resolve_utf8_locale() -> str:
    """选一个本机已安装的 UTF-8 locale。

    硬编码 zh_CN.UTF-8 在未安装该 locale 的机器上会让 LC_CTYPE 失效，
    X11/Qt 剪贴板字符集转换失败，粘贴中文就会乱码。
    """
    import locale as _locale

    candidates = (
        "zh_CN.UTF-8",
        "zh_CN.utf8",
        "C.UTF-8",
        "C.utf8",
        "en_US.UTF-8",
        "en_US.utf8",
    )
    for name in candidates:
        try:
            _locale.setlocale(_locale.LC_ALL, name)
            return name
        except _locale.Error:
            continue
    # 最后兜底：保持进程当前可用 locale（通常已是 UTF-8）
    try:
        cur = _locale.setlocale(_locale.LC_ALL, "")
        if cur:
            return cur
    except _locale.Error:
        pass
    return "C.UTF-8"


def configure_qt_ime_for_chinese() -> None:
    """为 Docker/本机启用 fcitx 中文输入（须在创建 QApplication 之前调用）。"""
    _eai_platform_bootstrap.os.environ.setdefault("QT_X11_NO_MITSHM", "1")
    utf8_loc = _resolve_utf8_locale()
    _eai_platform_bootstrap.os.environ["LANG"] = utf8_loc
    _eai_platform_bootstrap.os.environ["LC_ALL"] = utf8_loc
    _eai_platform_bootstrap.os.environ["QT_IM_MODULE"] = "fcitx"
    _eai_platform_bootstrap.os.environ["XMODIFIERS"] = "@im=fcitx"
    _eai_platform_bootstrap.os.environ["GTK_IM_MODULE"] = "fcitx"

    # PyQt5 自带 Qt 库优先，避免 fcitx 插件与系统 Qt 混库
    try:
        import PyQt5

        qt_lib = _eai_platform_bootstrap.os.path.join(
            _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.abspath(PyQt5.__file__)), "Qt5", "lib"
        )
        if _eai_platform_bootstrap.os.path.isdir(qt_lib):
            cur = _eai_platform_bootstrap.os.environ.get("LD_LIBRARY_PATH", "")
            parts = [p for p in cur.split(":") if p and p != qt_lib]
            _eai_platform_bootstrap.os.environ["LD_LIBRARY_PATH"] = ":".join([qt_lib] + parts)
    except Exception:
        pass

    try:
        import PyQt5
        from PyQt5.QtCore import QCoreApplication

        plug = _eai_platform_bootstrap.os.path.join(
            _eai_platform_bootstrap.os.path.dirname(_eai_platform_bootstrap.os.path.abspath(PyQt5.__file__)),
            "Qt5",
            "plugins",
        )
        if _eai_platform_bootstrap.os.path.isdir(plug):
            QCoreApplication.addLibraryPath(plug)
    except Exception:
        pass

