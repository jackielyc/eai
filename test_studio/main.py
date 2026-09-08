#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立启动测试工作室前端。

用法（容器内，与 run_in_docker 相同环境）::

    python3 -m test_studio.main

或宿主机 / Docker::

    bash test_studio/run_test_studio.sh
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    eai_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if eai_dir not in sys.path:
        sys.path.insert(0, eai_dir)

    # 先加载 show_camera_topics（提供 ChatPanel / 主题 / IME），再开窗口
    import show_camera_topics as sct
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from test_studio.window import TestStudioWindow

    sct.configure_qt_ime_for_chinese()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    sct.apply_viewer_theme(app)
    sct.install_chinese_ime_guards(app)
    app.setQuitOnLastWindowClosed(True)

    llm_config = sct.LlmChatConfig.from_env()
    window = TestStudioWindow(llm_config=llm_config)
    window.setAttribute(Qt.WA_QuitOnClose, True)
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
