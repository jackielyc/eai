"""application.entrypoint — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_adapters_ros_bridges = module_ref('eai_app.adapters.ros.bridges')
_eai_adapters_ros_camera_node = module_ref('eai_app.adapters.ros.camera_node')
_eai_application_cli = module_ref('eai_app.application.cli')
_eai_application_main_window = module_ref('eai_app.application.main_window')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_platform_qt_environment = module_ref('eai_app.platform.qt_environment')
_eai_ui_theme = module_ref('eai_app.ui.theme')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

def main(
    existing_app: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QApplication] = None,
    splash: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None,
) -> int:
    def _splash(msg: str) -> None:
        if splash is None:
            return
        try:
            label = getattr(splash, "_boot_status", None)
            t0 = getattr(splash, "_boot_t0", None)
            if label is not None:
                if t0 is not None:
                    label.setText(f"{msg}  ({_eai_platform_bootstrap.time.perf_counter() - t0:.1f}s)")
                else:
                    label.setText(msg)
            app_inst = _eai_platform_bootstrap.QApplication.instance()
            if app_inst is not None:
                app_inst.processEvents()
        except Exception:
            pass

    pass
    _eai_platform_bootstrap._boot_progress = _splash if splash is not None else None

    args = _eai_application_cli.parse_args()
    only_tabs = _eai_application_cli.parse_only_tabs(args.tab)

    if existing_app is None:
        _eai_platform_qt_environment.configure_qt_ime_for_chinese()
        _eai_platform_qt_environment.configure_qt_webengine_for_remote_display()
        # QWebEngineView 需要在创建 QApplication 前设置
        from PyQt5.QtCore import QCoreApplication

        QCoreApplication.setAttribute(_eai_platform_bootstrap.Qt.AA_ShareOpenGLContexts, True)
        app = _eai_platform_bootstrap.QApplication(_eai_platform_bootstrap.sys.argv)
    else:
        app = existing_app

    app.setStyle("Fusion")
    _eai_ui_theme.apply_viewer_theme(app)
    _eai_ui_widgets_inputs.install_chinese_ime_guards(app)
    app.setQuitOnLastWindowClosed(True)

    _splash("加载 OpenCV…")
    _eai_platform_bootstrap._ensure_cv2_numpy()

    _splash("初始化 ROS…")
    _eai_platform_bootstrap.rclpy.init()

    bridge = _eai_adapters_ros_bridges.RosBridge()
    node = _eai_adapters_ros_camera_node.CameraTopicNode(bridge, prefix=args.prefix)
    llm_config = _eai_adapters_models_chat_client.LlmChatConfig(
        api_base=args.llm_api_base,
        model=args.llm_model,
        api_key=_eai_platform_bootstrap.os.environ.get(_eai_configuration_chat.LLM_API_KEY_ENV, "").strip(),
    )

    _splash("构建主界面…")
    window = _eai_application_main_window.CameraTopicWindow(
        node,
        bridge,
        prefix=args.prefix,
        llm_config=llm_config,
        only_tabs=only_tabs or None,
    )
    window.setAttribute(_eai_platform_bootstrap.Qt.WA_QuitOnClose, True)
    window.show()
    if splash is not None:
        try:
            splash.close()
            splash.deleteLater()
        except Exception:
            pass
    app.processEvents()
    window.build_heavy_ui()
    _eai_platform_bootstrap._boot_progress = None
    # UI 重启后若共享 .npy 仍在，自动恢复图像预览（不依赖本窗口是否拥有评测进程）
    _eai_platform_bootstrap.QTimer.singleShot(800, window._maybe_resume_sim_npy_preview)
    # 勿在后台线程预热 OpenGL：Qt/GL context 非主线程初始化易 SIP segfault。
    # 首次点「启用 3D」时再 import pyqtgraph.opengl。

    shutdown_flag = {"value": False}
    cleaned_up = {"value": False}

    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    # 单线程 executor：少和 Qt 抢 GIL（关节 topic 高频时 4 线程更卡）
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def cleanup() -> None:
        if cleaned_up["value"]:
            return
        cleaned_up["value"] = True
        shutdown_flag["value"] = True
        sig_timer.stop()
        try:
            node.prepare_shutdown()
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        ros_thread.join(timeout=1.0)
        if _eai_platform_bootstrap.rclpy.ok():
            try:
                node.destroy_node()
            except Exception:
                pass
            try:
                _eai_platform_bootstrap.rclpy.shutdown()
            except Exception:
                pass

    def ros_spin_wrapper() -> None:
        while _eai_platform_bootstrap.rclpy.ok() and not shutdown_flag["value"]:
            try:
                try:
                    node.drain_ros_ops(max_ops=1)
                except Exception:
                    pass
                try:
                    node.maybe_pump_robot_extras()
                except Exception:
                    pass
                executor.spin_once(timeout_sec=0.05)
            except ExternalShutdownException:
                break
            except Exception:
                if shutdown_flag["value"]:
                    break

    ros_thread = _eai_platform_bootstrap.threading.Thread(target=ros_spin_wrapper, daemon=True)
    ros_thread.start()

    def on_sigint(*_args) -> None:
        if shutdown_flag["value"]:
            _eai_platform_bootstrap.os._exit(130)
        app.quit()

    _eai_platform_bootstrap.signal.signal(_eai_platform_bootstrap.signal.SIGINT, on_sigint)

    # Qt 事件循环会阻塞 Python 信号处理，定时器让 Ctrl+C 能及时生效
    sig_timer = _eai_platform_bootstrap.QTimer(app)
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start(50)

    def _persist_viewer_ui() -> None:
        try:
            window._save_viewer_ui_state()
        except Exception as exc:
            print(f"[viewer_ui_state] save failed: {exc}", flush=True)

    app.aboutToQuit.connect(_persist_viewer_ui)
    app.aboutToQuit.connect(cleanup)

    result = 0
    try:
        result = app.exec_()
    except KeyboardInterrupt:
        app.quit()
    finally:
        _persist_viewer_ui()
        cleanup()
    return result

