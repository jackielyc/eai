"""ui.window.shell open local cam preview source — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_adapters_models_qwen_remote = module_ref('eai_app.adapters.models.qwen_remote')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

class ShellOpenLocalCamPreviewSourceMixin:
    def _open_local_cam_preview_source(self, url_or_dev: str) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """打开网络流 URL，或本机 /dev/video* / 数字 index。"""
        src = (url_or_dev or "").strip()
        if not src:
            return False, "请填写预览 URL 或本机设备（例如 4 或 /dev/video4）"
        self._close_local_cam_preview()
        cap = None
        try:
            is_url = "://" in src or src.lower().endswith((".mjpg", ".mjpeg"))
            if is_url:
                cap = _eai_platform_bootstrap.cv2.VideoCapture(src, _eai_platform_bootstrap.cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = _eai_platform_bootstrap.cv2.VideoCapture(src)
            else:
                index: object = src
                if src.isdigit():
                    index = int(src)
                elif src.startswith("/dev/video") and src[len("/dev/video") :].isdigit():
                    index = int(src[len("/dev/video") :])
                if isinstance(index, int):
                    cap = _eai_platform_bootstrap.cv2.VideoCapture(index, _eai_platform_bootstrap.cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap.release()
                        cap = _eai_platform_bootstrap.cv2.VideoCapture(index)
                else:
                    cap = _eai_platform_bootstrap.cv2.VideoCapture(src)
            if not cap.isOpened():
                if is_url:
                    hint = (
                        f"无法打开:\n{src}\n\n"
                        "请先「启动推流」与隧道，或改用上方「打开本机监控」。"
                    )
                else:
                    hint = (
                        f"无法打开本机设备:\n{src}\n\n"
                        "可点「刷新设备」查看 /dev/video*；"
                        "RealSense 彩色节点通常是 video4。"
                    )
                return False, hint
            try:
                cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            deadline = _eai_platform_bootstrap.time.time() + (8.0 if is_url else 2.0)
            ok = False
            frame = None
            while _eai_platform_bootstrap.time.time() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    break
                _eai_platform_bootstrap.time.sleep(0.05)
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                cap.release()
                return False, f"已连接但读不到画面:\n{src}"
            self._local_cam_preview_cap = cap
            return True, src
        except Exception as exc:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return False, f"打开预览失败: {exc}"

    def _start_local_cam_preview_ui(self, source: str, mode: str) -> None:
        self._local_cam_preview_active = True
        self._local_cam_preview_mode = mode
        short = source if len(source) <= 56 else source[:53] + "..."
        if mode == "device":
            if hasattr(self, "local_cam_monitor_btn"):
                self.local_cam_monitor_btn.setText("停止本机监控")
            if hasattr(self, "local_cam_monitor_status"):
                self.local_cam_monitor_status.setText(f"监控: 播放中 @ {short}")
                self.local_cam_monitor_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            if hasattr(self, "local_cam_preview_status"):
                self.local_cam_preview_status.setText(f"预览: 本机设备 {short}")
                self.local_cam_preview_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self._set_local_cam_device_controls_enabled(False)
            if hasattr(self, "local_cam_url_edit"):
                self.local_cam_url_edit.setEnabled(True)
            if hasattr(self, "local_cam_preview_btn"):
                self.local_cam_preview_btn.setText("打开预览")
        else:
            if hasattr(self, "local_cam_url_edit"):
                self.local_cam_url_edit.setEnabled(False)
            if hasattr(self, "local_cam_preview_btn"):
                self.local_cam_preview_btn.setText("停止预览")
            if hasattr(self, "local_cam_preview_status"):
                self.local_cam_preview_status.setText(f"预览: 播放中 @ {short}")
                self.local_cam_preview_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            if hasattr(self, "local_cam_monitor_btn"):
                self.local_cam_monitor_btn.setText("打开本机监控")
            if hasattr(self, "local_cam_monitor_status"):
                self.local_cam_monitor_status.setText("监控: 未开启（当前为网络预览）")
                self.local_cam_monitor_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        if hasattr(self, "local_cam_preview_label"):
            self.local_cam_preview_label.setText("")
        self._local_cam_preview_timer.start()

    def _auto_open_local_cam_monitor(self) -> None:
        """已禁用自动打开（避免启动卡在 VideoCapture）。"""
        return

    def _ensure_local_cam_monitor_open(self, *, silent: bool = False) -> bool:
        """打开本机直连监控；已在 device 模式则直接成功。silent 时失败不弹窗。"""
        if self._local_cam_preview_active and getattr(
            self, "_local_cam_preview_mode", ""
        ) == "device":
            return True
        if self._local_cam_preview_active:
            self._stop_local_cam_preview()
        device = self.local_cam_device_edit.text().strip() or "0"
        self.local_cam_device_edit.setText(device)
        if hasattr(self, "local_cam_monitor_status"):
            self.local_cam_monitor_status.setText(f"监控: 连接中… {device}")
            self.local_cam_monitor_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        self.status_bar.showMessage(f"正在打开本机摄像头: {device}")
        _eai_platform_bootstrap.QApplication.processEvents()
        ok, msg = self._open_local_cam_preview_source(device)
        if not ok:
            # 不再自动扫其它 /dev/video*（OpenCV 探测会卡数秒～假死）
            if not silent:
                _eai_platform_bootstrap.QMessageBox.warning(self, "本机摄像头监控", msg)
            else:
                self._append_local_cam_log(f"[监控] 打开失败: {msg}")
            if hasattr(self, "local_cam_monitor_status"):
                self.local_cam_monitor_status.setText("监控: 连接失败")
                self.local_cam_monitor_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            return False
        self._start_local_cam_preview_ui(msg, "device")
        self._append_local_cam_log(f"[监控] 已开启: {msg}")
        self.status_bar.showMessage(f"本机摄像头监控已开启: {msg}", 4000)
        return True

    def _apply_local_cam_rotation(self, frame: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.np.ndarray:
        deg = int(getattr(self, "_local_cam_rotate_deg", 0) or 0) % 360
        if deg == 90:
            return _eai_platform_bootstrap.cv2.rotate(frame, _eai_platform_bootstrap.cv2.ROTATE_90_CLOCKWISE)
        if deg == 180:
            return _eai_platform_bootstrap.cv2.rotate(frame, _eai_platform_bootstrap.cv2.ROTATE_180)
        if deg == 270:
            return _eai_platform_bootstrap.cv2.rotate(frame, _eai_platform_bootstrap.cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _update_local_cam_rotate_btn(self) -> None:
        if not hasattr(self, "local_cam_rotate_btn"):
            return
        deg = int(getattr(self, "_local_cam_rotate_deg", 0) or 0) % 360
        self.local_cam_rotate_btn.setText(f"旋转 {deg}°")

    def _init_qwen_deploy_controller(self) -> None:
        """测试 Tab 挂载共享推理控制器（测试工作室为独立窗口，见 test_studio/run_test_studio.sh）。"""
        if _eai_configuration_models.EAI_DIR not in _eai_platform_bootstrap.sys.path:
            _eai_platform_bootstrap.sys.path.insert(0, _eai_configuration_models.EAI_DIR)
        from test_studio import DeployView, QwenDeployController

        self._qwen_deploy = QwenDeployController(self)
        self._qwen_deploy.status_message.connect(self.status_bar.showMessage)
        self._local_qwen_launcher = self._qwen_deploy.local_launcher
        self._remote_qwen_launcher = self._qwen_deploy.remote_launcher
        self._test_qwen_model_list_refreshing = False
        self._pending_test_qwen_model_path = ""

        legacy_view = DeployView(
            target_combo=self.test_qwen_target_combo,
            root_combo=self.test_qwen_root_combo,
            model_combo=self.test_qwen_model_combo,
            refresh_btn=self.test_qwen_refresh_btn,
            load_last_btn=self.test_qwen_load_last_btn,
            start_btn=self.test_qwen_start_btn,
            stop_btn=self.test_qwen_stop_btn,
            path_label=self.test_model_path_label,
            status_label=self.test_qwen_status_label,
            log_edit=self.test_infer_log_edit,
            show_message=self.status_bar.showMessage,
            get_chat_panel=lambda: self.chat_panel,
            focus_host=self,
        )
        self._qwen_deploy.register_view(legacy_view)

    def _on_open_test_studio_clicked(self) -> None:
        """从「测试」Tab 拉起独立的测试工作室进程（不嵌入本窗口）。"""
        try:
            # 已在容器内时直接起独立进程；宿主机则走 test_studio/run_test_studio.sh
            in_docker = _eai_platform_bootstrap.os.path.exists("/.dockerenv") or _eai_platform_bootstrap.os.environ.get(
                "A2D_IN_DOCKER", ""
            ).strip() in ("1", "true", "yes")
            if in_docker:
                env = _eai_platform_bootstrap.os.environ.copy()
                env["PYTHONPATH"] = (
                    _eai_configuration_models.EAI_DIR + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                )
                _eai_platform_bootstrap.subprocess.Popen(
                    [_eai_platform_bootstrap.sys.executable, "-m", "test_studio.main"],
                    cwd=_eai_configuration_models.EAI_DIR,
                    env=env,
                    start_new_session=True,
                )
            else:
                script = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, "test_studio", "run_test_studio.sh")
                if not _eai_platform_bootstrap.os.path.isfile(script):
                    self.status_bar.showMessage(f"未找到: {script}")
                    return
                _eai_platform_bootstrap.subprocess.Popen(
                    ["bash", script],
                    cwd=_eai_configuration_models.EAI_DIR,
                    start_new_session=True,
                )
            self.status_bar.showMessage("已启动独立「测试工作室」窗口…")
        except Exception as exc:
            self.status_bar.showMessage(f"启动测试工作室失败: {exc}")

    def _on_test_scenario_clicked(self, title: str) -> None:
        path = self._test_image_paths.get(title, "")
        if not path:
            self.status_bar.showMessage(f"场景「{title}」无可用图片")
            return
        if self.chat_panel.set_attached_image_from_path(path, display_name=title):
            self._set_test_scenario_selection(title)
            self.status_bar.showMessage(f"已选场景图用于对话: {title}")

    def _set_test_scenario_selection(self, title: str) -> None:
        self._test_selected_scenario = title
        for name, label in self._test_image_labels.items():
            label.set_selected(name == title)

    def _toggle_workspace_panel(self) -> None:
        """折叠 / 展开下方工作区（Topic + 图像预览 + AI 对话）。"""
        splitter = getattr(self, "_ui_vsplitter", None)
        if not self._workspace_collapsed:
            if splitter is not None:
                sizes = splitter.sizes()
                if len(sizes) == 2 and sizes[1] > 48:
                    self._ui_vsplitter_saved_sizes = list(sizes)
            self._workspace_body.hide()
            self._workspace_title_label.setText("工作区已折叠")
            self._workspace_toggle_btn.setText("展开")
            self._workspace_toggle_btn.setToolTip("展开 Topic / 图像预览 / AI 对话")
            self._workspace_panel.setMaximumHeight(40)
            self._workspace_panel.setMinimumHeight(36)
            self._workspace_panel.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Fixed)
            self._workspace_collapsed = True
            if splitter is not None:
                total = max(1, sum(splitter.sizes()) or splitter.height())
                splitter.setSizes([max(200, total - 40), 40])
            if getattr(self, "status_bar", None) is not None:
                self.status_bar.showMessage("已折叠工作区", 2000)
        else:
            self._workspace_body.show()
            self._workspace_title_label.setText("Topic / 图像预览 / Live HUD / AI 对话")
            self._workspace_toggle_btn.setText("折叠")
            self._workspace_toggle_btn.setToolTip(
                "折叠下方工作区（Topic、图像预览、Live HUD、AI 对话）"
            )
            self._workspace_panel.setMaximumHeight(16777215)
            self._workspace_panel.setMinimumHeight(160)
            self._workspace_panel.setSizePolicy(
                _eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding
            )
            self._workspace_collapsed = False
            if splitter is not None:
                saved = getattr(self, "_ui_vsplitter_saved_sizes", None)
                if saved and len(saved) == 2 and saved[1] >= 160:
                    splitter.setSizes(saved)
                else:
                    total = max(1, sum(splitter.sizes()) or splitter.height())
                    splitter.setSizes(
                        [max(220, int(total * 0.55)), max(200, int(total * 0.45))]
                    )
            if getattr(self, "status_bar", None) is not None:
                self.status_bar.showMessage("已展开工作区", 2000)

    def _append_test_infer_log(self, line: str) -> None:
        if getattr(self, "_qwen_deploy", None) is not None:
            self._qwen_deploy.append_log(line)
            return
        self.test_infer_log_edit.append(line)
        scrollbar = self.test_infer_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _selected_test_qwen_deploy_spec(self) -> _eai_platform_bootstrap.Dict[str, str]:
        data = self.test_qwen_model_combo.currentData()
        if isinstance(data, dict):
            path = str(data.get("path") or "").strip()
            if path:
                return {
                    "path": path,
                    "model_id": str(data.get("model_id") or _eai_platform_bootstrap.os.path.basename(path.rstrip("/"))),
                    "label": str(data.get("label") or data.get("name") or _eai_platform_bootstrap.os.path.basename(path.rstrip("/"))),
                    "kind": str(data.get("kind") or ""),
                    "root": str(data.get("root") or ""),
                }
        return {}

    def _selected_test_qwen_model_key(self) -> str:
        spec = self._selected_test_qwen_deploy_spec()
        if spec.get("path"):
            return str(spec.get("model_id") or spec.get("path"))
        key = self.test_qwen_model_combo.currentData()
        if isinstance(key, str) and key:
            return key
        return _eai_configuration_policies.LOCAL_QWEN_DEPLOY_MODELS[0][0]

    def _selected_test_qwen_scan_root(self) -> str:
        root = self.test_qwen_root_combo.currentData()
        if isinstance(root, str) and root.strip():
            return root.strip()
        return ""

    def _refresh_test_qwen_scan_roots(self) -> None:
        target = self._selected_test_qwen_target()
        prev = self._selected_test_qwen_scan_root()
        self.test_qwen_root_combo.blockSignals(True)
        self.test_qwen_root_combo.clear()
        if target == "local":
            roots = _eai_adapters_models_qwen_local.local_deploy_scan_roots()
        else:
            roots = _eai_adapters_models_qwen_local.remote_deploy_scan_roots(self._selected_test_qwen_remote_host())
        for path, label in roots:
            self.test_qwen_root_combo.addItem(label, path)
            idx = self.test_qwen_root_combo.count() - 1
            self.test_qwen_root_combo.setItemData(idx, path, _eai_platform_bootstrap.Qt.ToolTipRole)
        if prev:
            idx = self.test_qwen_root_combo.findData(prev)
            if idx >= 0:
                self.test_qwen_root_combo.setCurrentIndex(idx)
        self.test_qwen_root_combo.blockSignals(False)

    def _on_test_qwen_root_changed(self, _index: int = 0) -> None:
        self._refresh_test_qwen_model_list()

    def _refresh_test_qwen_model_list(self) -> None:
        if self._test_qwen_model_list_refreshing:
            return
        root = self._selected_test_qwen_scan_root()
        if not root:
            return
        self._test_qwen_model_list_refreshing = True
        # 禁用前把焦点还给文本框，避免 focused+disabled 弄死 fcitx
        fw = _eai_platform_bootstrap.QApplication.focusWidget()
        if fw is self.test_qwen_model_combo or (
            fw is not None and self.test_qwen_model_combo.isAncestorOf(fw)
        ):
            target = _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET
            if target is not None and _eai_ui_widgets_inputs._is_text_ime_widget(target):
                target.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)
            else:
                self.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)
        self.test_qwen_refresh_btn.setEnabled(False)
        self.test_qwen_model_combo.setEnabled(False)
        _eai_ui_widgets_inputs._schedule_fcitx_restore(_eai_platform_bootstrap._IME_LAST_TEXT_WIDGET)
        target = self._selected_test_qwen_target()
        host_id = self._selected_test_qwen_remote_host()
        self._append_test_infer_log(f"扫描可部署目录: {root} …")

        def _work() -> None:
            ok, models, err = _eai_adapters_models_qwen_local.fetch_deploy_model_dirs(
                target=target,
                host_id=host_id,
                root=root,
            )
            self._deploy_model_list_bridge.finished.emit(
                {
                    "ok": ok,
                    "models": models,
                    "message": err,
                    "root": root,
                    "target": target,
                }
            )

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_test_qwen_model_list_ready(self, payload: object) -> None:
        self._test_qwen_model_list_refreshing = False
        data = payload if isinstance(payload, dict) else {}
        ok = bool(data.get("ok"))
        models = data.get("models") if isinstance(data.get("models"), list) else []
        root = str(data.get("root") or self._selected_test_qwen_scan_root())
        prev_path = (
            self._pending_test_qwen_model_path
            or str(self._selected_test_qwen_deploy_spec().get("path") or "")
        )
        self._pending_test_qwen_model_path = ""

        self.test_qwen_model_combo.blockSignals(True)
        self.test_qwen_model_combo.clear()
        for entry in models:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            if not path:
                continue
            name = str(entry.get("name") or _eai_platform_bootstrap.os.path.basename(path.rstrip("/")))
            kind = str(entry.get("kind") or "")
            kind_label = _eai_adapters_models_qwen_local.deploy_model_kind_label(kind)
            readable = str(entry.get("readable") or "1") != "0"
            item_label = f"{name} ({kind_label})"
            if not readable:
                item_label = f"{name} ({kind_label} · 不可读)"
            spec = {
                "path": path,
                "name": name,
                "model_id": str(entry.get("model_id") or name),
                "label": str(entry.get("label") or name),
                "kind": kind,
                "root": str(entry.get("root") or root),
                "readable": "1" if readable else "0",
            }
            self.test_qwen_model_combo.addItem(item_label, spec)
            idx = self.test_qwen_model_combo.count() - 1
            tip = (
                f"{name}\nmodel_id={spec['model_id']}\n"
                f"路径: {path}\n类型: {kind_label}\n根目录: {spec['root']}"
            )
            if not readable:
                tip += (
                    "\n权重不可读（多为 root 600）："
                    "sudo chmod a+r <dir>/adapter_model.safetensors"
                )
                model = self.test_qwen_model_combo.model()
                item = model.item(idx) if hasattr(model, "item") else None
                if item is not None:
                    item.setEnabled(False)
            self.test_qwen_model_combo.setItemData(idx, tip, _eai_platform_bootstrap.Qt.ToolTipRole)
        if prev_path:
            for i in range(self.test_qwen_model_combo.count()):
                spec = self.test_qwen_model_combo.itemData(i)
                if isinstance(spec, dict) and spec.get("path") == prev_path:
                    self.test_qwen_model_combo.setCurrentIndex(i)
                    break
        elif self.test_qwen_model_combo.count() > 0:
            prefer = self.test_qwen_model_combo.findText(
                "Qwen3.5-35B-A3B", _eai_platform_bootstrap.Qt.MatchContains
            )
            self.test_qwen_model_combo.setCurrentIndex(
                prefer if prefer >= 0 else 0
            )
        self.test_qwen_model_combo.blockSignals(False)

        if ok:
            self._append_test_infer_log(
                f"已加载 {len(models)} 个可部署目录（{root}）"
            )
        else:
            msg = str(data.get("message") or "扫描失败")
            self._append_test_infer_log(f"目录扫描失败: {msg}")
            self.status_bar.showMessage(f"目录扫描失败: {msg}")
        self._refresh_test_qwen_model_path_label()
        self._update_test_qwen_ui()

    def _selected_test_qwen_target(self) -> str:
        target = str(self.test_qwen_target_combo.currentData() or "local")
        if target == "local":
            return "local"
        if target.startswith("remote:") or target == "remote":
            return "remote"
        return "local"

    def _selected_test_qwen_remote_host(self) -> str:
        target = str(self.test_qwen_target_combo.currentData() or "")
        if target.startswith("remote:"):
            return target.split(":", 1)[1].strip() or _eai_configuration_policies.REMOTE_QWEN_SSH_HOST
        return _eai_configuration_policies.REMOTE_QWEN_SSH_HOST

    def _on_test_qwen_target_changed(self, _index: int = 0) -> None:
        if self._selected_test_qwen_target() == "remote":
            self._remote_qwen_launcher.set_host(self._selected_test_qwen_remote_host())
        self._refresh_test_qwen_scan_roots()
        self._refresh_test_qwen_model_list()
        self._update_test_qwen_ui()

    def _on_test_qwen_model_changed(self, _index: int = 0) -> None:
        self._refresh_test_qwen_model_path_label()

    def _refresh_test_qwen_model_path_label(self) -> None:
        spec = self._selected_test_qwen_deploy_spec()
        target = self._selected_test_qwen_target()
        if spec.get("path"):
            path = spec["path"]
            label = spec.get("label") or _eai_platform_bootstrap.os.path.basename(path.rstrip("/"))
            mid = spec.get("model_id") or label
            kind = _eai_adapters_models_qwen_local.deploy_model_kind_label(str(spec.get("kind") or ""))
            if target == "remote":
                host_id = self._selected_test_qwen_remote_host()
                prof = _eai_adapters_models_qwen_remote.remote_qwen_profile(host_id)
                py = _eai_adapters_models_qwen_local.resolve_deploy_python_for_spec(
                    spec, target="remote", host_id=host_id
                )
                api = str(prof.get("api_base") or _eai_adapters_models_qwen_remote.remote_qwen_api_base_for_host(host_id))
                self.test_model_path_label.setText(
                    f"{host_id}:{path}  ·  id={mid}"
                )
                self.test_model_path_label.setStyleSheet(
                    f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};"
                )
                self.test_model_path_label.setToolTip(
                    f"远程部署 {label}（{kind}）\nSSH Host: {host_id}\n"
                    f"model_id={mid}\n路径: {path}\n"
                    f"Python: {py}\n"
                    f"工作目录: {prof.get('remote_work')}\n"
                    f"隧道 API: {api}"
                )
            else:
                self.test_model_path_label.setText(f"{path}  ·  id={mid}")
                self.test_model_path_label.setStyleSheet(
                    f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};"
                )
                self.test_model_path_label.setToolTip(
                    f"本地部署 {label}（{kind}）\nmodel_id={mid}\n"
                    f"路径: {path}\n"
                    f"API: {_eai_configuration_policies.LOCAL_QWEN_API_BASE_DEFAULT}"
                )
            return

        key = self._selected_test_qwen_model_key()
        label = _eai_adapters_models_qwen_local.local_qwen_label_for_key(key)
        mid = _eai_adapters_models_qwen_local.local_qwen_model_id_for_key(key)
        dirname = _eai_adapters_models_qwen_local.qwen_deploy_path_spec_for_key(key)
        if target == "remote":
            host_id = self._selected_test_qwen_remote_host()
            prof = _eai_adapters_models_qwen_remote.remote_qwen_profile(host_id)
            remote_root = str(prof.get("model_root") or "")
            remote_path = _eai_adapters_models_qwen_local.qwen_model_path_from_spec(dirname, root=remote_root)
            py = _eai_adapters_models_qwen_local.remote_qwen_python_for_key(key) or str(prof.get("python") or "")
            api = str(prof.get("api_base") or _eai_adapters_models_qwen_remote.remote_qwen_api_base_for_host(host_id))
            kind = "LoRA" if _eai_adapters_models_qwen_local.qwen_model_is_lora(remote_path) else "全量"
            self.test_model_path_label.setText(
                f"{host_id}:{remote_path}  ·  id={mid}"
            )
            self.test_model_path_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
            self.test_model_path_label.setToolTip(
                f"远程部署 {label}（{kind}）\nSSH Host: {host_id}\n"
                f"model_id={mid}\n路径: {remote_path}\n"
                f"Python: {py}\n"
                f"工作目录: {prof.get('remote_work')}\n"
                f"隧道 API: {api}"
            )
            return

        path = _eai_adapters_models_qwen_local.local_qwen_model_dir_for_key(key)
        missing = _eai_adapters_models_qwen_local.qwen_model_path_from_spec(dirname)
        if path:
            self.test_model_path_label.setText(f"{path}  ·  id={mid}")
            self.test_model_path_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        else:
            self.test_model_path_label.setText(f"未找到权重: {missing}")
            self.test_model_path_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        self.test_model_path_label.setToolTip(
            f"本地部署 {label}\nmodel_id={mid}\n"
            f"路径: {path or missing}\n"
            f"API: {_eai_configuration_policies.LOCAL_QWEN_API_BASE_DEFAULT}"
        )

    def _on_test_qwen_status(self, msg: str) -> None:
        self.test_qwen_status_label.setText(f"服务: {msg}")
        self._append_test_infer_log(msg)
        self.status_bar.showMessage(msg)

    def _refresh_test_qwen_status(self) -> None:
        self._update_test_qwen_ui()

    def _update_test_qwen_ui(self, *_args) -> None:
        if getattr(self, "_qwen_deploy", None) is not None:
            self._qwen_deploy.update_all_ui()
            return
        target = self._selected_test_qwen_target()
        if target == "remote":
            self._update_test_qwen_ui_remote()
        else:
            self._update_test_qwen_ui_local()

    def _update_test_qwen_ui_remote(self) -> None:
        host_id = self._selected_test_qwen_remote_host()
        self._remote_qwen_launcher.set_host(host_id)
        api = self._remote_qwen_launcher.api_base()
        info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api)
        healthy = bool(info and info.get("ok"))
        was_starting = bool(getattr(self._remote_qwen_launcher, "_starting", False))
        starting = was_starting and not healthy
        if healthy:
            self._remote_qwen_launcher._starting = False
            starting = False
            if was_starting:
                mid = str(
                    (info or {}).get("model")
                    or self._remote_qwen_launcher.last_model_id()
                )
                self.chat_panel.apply_remote_qwen_service_preset(
                    api_base=api, model_id=mid, silent=False
                )
                self._append_test_infer_log(
                    f"远程 Qwen 已就绪: {api} ({mid})"
                    f"{_eai_adapters_models_qwen_local.qwen_deploy_mode_suffix(info)}"
                )
        live_model = str((info or {}).get("model") or "")
        if healthy:
            suffix = f" · {live_model}" if live_model else ""
            suffix += _eai_adapters_models_qwen_local.qwen_deploy_mode_suffix(info)
            self.test_qwen_status_label.setText(f"服务: 远程在线 {api}{suffix}")
            self.test_qwen_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        elif starting:
            label = self._remote_qwen_launcher.last_model_label()
            self.test_qwen_status_label.setText(
                f"服务: 远程启动中（{label} @ {host_id}）…"
            )
            self.test_qwen_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        else:
            self.test_qwen_status_label.setText(f"服务: 远程离线（{host_id}）")
            self.test_qwen_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        self.test_qwen_start_btn.setEnabled(not healthy and not starting)
        self.test_qwen_stop_btn.setEnabled(healthy or starting)
        service_busy = healthy or starting
        scanning = bool(self._test_qwen_model_list_refreshing)
        # 仅扫描目录时保持「部署位置/权重目录」可点
        self.test_qwen_model_combo.setEnabled(not service_busy and not scanning)
        self.test_qwen_target_combo.setEnabled(not service_busy)
        self.test_qwen_root_combo.setEnabled(not service_busy)
        self.test_qwen_refresh_btn.setEnabled(not service_busy and not scanning)
        self.test_qwen_load_last_btn.setEnabled(not service_busy and not scanning)

    def _update_test_qwen_ui_local(self) -> None:
        api = self._local_qwen_launcher.api_base()
        info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api)
        healthy = bool(info and info.get("ok"))
        hostctl_running = False
        hostctl_starting = False
        hostctl_error = ""
        if _eai_adapters_models_qwen_local.check_local_qwen_hostctl_health():
            _ok, body = _eai_adapters_models_qwen_local.local_qwen_hostctl_request("/status", method="GET", timeout_s=2.0)
            if _ok:
                hostctl_running = bool(body.get("process_running"))
                hostctl_starting = bool(body.get("starting")) or (
                    hostctl_running and not healthy
                )
                hostctl_error = str(body.get("last_error") or "").strip()

        if healthy:
            self._local_qwen_launcher._starting = False
        elif getattr(self._local_qwen_launcher, "_starting", False):
            local_proc_alive = (
                self._local_qwen_launcher._process is not None
                and self._local_qwen_launcher._process.state() == _eai_platform_bootstrap.QProcess.Running
            )
            if self._local_qwen_launcher._via_hostctl:
                if not hostctl_running and not healthy:
                    self._local_qwen_launcher._starting = False
                    self._local_qwen_launcher._via_hostctl = False
                    if not self._local_qwen_launcher._fail_notified:
                        self._local_qwen_launcher._fail_notified = True
                        err = hostctl_error or (
                            "推理进程已退出，请查看 log/local_qwen_service.log"
                        )
                        self._append_test_infer_log(f"启动失败: {err}")
                        self.status_bar.showMessage("本地 Qwen 启动失败")
            elif not local_proc_alive:
                self._local_qwen_launcher._starting = False

        starting = bool(getattr(self._local_qwen_launcher, "_starting", False)) and not healthy
        if (
            self._local_qwen_launcher._process is not None
            and self._local_qwen_launcher._process.state() == _eai_platform_bootstrap.QProcess.Running
            and not healthy
        ):
            starting = True
        if hostctl_starting:
            starting = True
            self._local_qwen_launcher._starting = True

        live_model = str((info or {}).get("model") or "")
        if healthy:
            suffix = f" · {live_model}" if live_model else ""
            suffix += _eai_adapters_models_qwen_local.qwen_deploy_mode_suffix(info)
            self.test_qwen_status_label.setText(f"服务: 在线 {api}{suffix}")
            self.test_qwen_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        elif starting:
            label = self._local_qwen_launcher.last_model_label()
            self.test_qwen_status_label.setText(
                f"服务: 启动中（加载 {label}，约 1–3 分钟）…"
            )
            self.test_qwen_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
            self.test_qwen_status_label.setToolTip(
                "正在向 GPU 加载权重；端口就绪前 / 加载完成前状态会保持「启动中」。\n"
                "日志: log/local_qwen_service.log"
            )
        elif hostctl_error and not healthy:
            short = hostctl_error if len(hostctl_error) <= 80 else hostctl_error[:77] + "…"
            self.test_qwen_status_label.setText(f"服务: 启动失败 · {short}")
            self.test_qwen_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            self.test_qwen_status_label.setToolTip(hostctl_error)
        else:
            self.test_qwen_status_label.setText("服务: 离线")
            self.test_qwen_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        self.test_qwen_start_btn.setEnabled(not healthy and not starting)
        self.test_qwen_stop_btn.setEnabled(healthy or starting)
        service_busy = healthy or starting
        scanning = bool(self._test_qwen_model_list_refreshing)
        self.test_qwen_model_combo.setEnabled(not service_busy and not scanning)
        self.test_qwen_target_combo.setEnabled(not service_busy)
        self.test_qwen_root_combo.setEnabled(not service_busy)
        self.test_qwen_refresh_btn.setEnabled(not service_busy and not scanning)
        self.test_qwen_load_last_btn.setEnabled(not service_busy and not scanning)

    def _save_test_qwen_deploy_config(self) -> None:
        spec = self._selected_test_qwen_deploy_spec()
        try:
            _eai_infrastructure_ui_storage.save_test_qwen_last_config(
                {
                    "target": str(self.test_qwen_target_combo.currentData() or "local"),
                    "scan_root": self._selected_test_qwen_scan_root(),
                    "model_path": str(spec.get("path") or ""),
                    "model_id": str(spec.get("model_id") or ""),
                    "model_label": str(spec.get("label") or ""),
                    "saved_at": _eai_adapters_models_chat_client._utc_now_iso(),
                }
            )
        except Exception as exc:
            self._append_test_infer_log(f"保存上次配置失败: {exc}")

    def _on_test_qwen_load_last_clicked(self) -> None:
        cfg = _eai_infrastructure_ui_storage.load_test_qwen_last_config()
        target = str(cfg.get("target") or "").strip()
        scan_root = str(cfg.get("scan_root") or "").strip()
        model_path = str(cfg.get("model_path") or "").strip()
        model_label = str(cfg.get("model_label") or model_path).strip()
        if not target and not scan_root and not model_path:
            self._append_test_infer_log("无上次模型配置（需先成功点过一次「启动推理服务」）")
            self.status_bar.showMessage("无上次推理配置")
            return

        self.test_qwen_target_combo.blockSignals(True)
        idx = self.test_qwen_target_combo.findData(target or "local")
        if idx >= 0:
            self.test_qwen_target_combo.setCurrentIndex(idx)
        self.test_qwen_target_combo.blockSignals(False)

        if self._selected_test_qwen_target() == "remote":
            self._remote_qwen_launcher.set_host(self._selected_test_qwen_remote_host())

        self._refresh_test_qwen_scan_roots()
        if scan_root:
            root_idx = self.test_qwen_root_combo.findData(scan_root)
            if root_idx >= 0:
                self.test_qwen_root_combo.setCurrentIndex(root_idx)
            else:
                self._append_test_infer_log(f"上次权重目录不在当前列表: {scan_root}")

        self._pending_test_qwen_model_path = model_path
        self._refresh_test_qwen_model_list()
        saved_at = str(cfg.get("saved_at") or "").strip()
        when = f" ({saved_at})" if saved_at else ""
        self._append_test_infer_log(
            f"已加载上次配置{when}: {target or 'local'} / {model_label or model_path or scan_root}"
        )
        self.status_bar.showMessage("已加载上次推理配置")

    def _on_test_qwen_start_clicked(self) -> None:
        spec = self._selected_test_qwen_deploy_spec()
        key = self._selected_test_qwen_model_key()
        target = self._selected_test_qwen_target()
        self._qwen_switch_tries = 0
        if not spec.get("path") and self.test_qwen_model_combo.count() == 0:
            self._append_test_infer_log("请先点「刷新」加载可部署目录，并选择一个模型。")
            self.status_bar.showMessage("未选择模型")
            return
        self._save_test_qwen_deploy_config()
        if target == "remote":
            host_id = self._selected_test_qwen_remote_host()
            path = spec.get("path") or ""
            if (
                path
                and not path.startswith("/share_data")
                and host_id
                in (
                    "tione-develop",
                    "nb-1668047664206989312-cp0nfah4b37k",
                )
            ):
                self._append_test_infer_log(
                    f"提示: 所选目录若不在 {host_id} 本机/共享盘，部署可能失败。"
                )
            api = _eai_adapters_models_qwen_remote.remote_qwen_api_base_for_host(host_id)
            label = spec.get("label") or key
            # Reuse healthy tunnel before re-deploy.
            if _eai_adapters_models_qwen_local.check_local_qwen_server_health(api):
                info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api) or {}
                mid = str(info.get("model") or label)
                self._append_test_infer_log(
                    f"远程 Qwen 已部署，跳过启动: {api} ({mid})"
                )
                self._remote_qwen_launcher.start(
                    model_key=key,
                    host_id=host_id,
                    model_dir=spec.get("path"),
                    model_id=spec.get("model_id"),
                    model_label=spec.get("label"),
                )
                self._update_test_qwen_ui()
                _eai_platform_bootstrap.QTimer.singleShot(500, self._try_switch_chat_to_remote_qwen)
                return
            self._append_test_infer_log(
                f"远程部署: {host_id} / {label}，隧道 {api}"
            )
            self._remote_qwen_launcher.start(
                model_key=key,
                host_id=host_id,
                model_dir=spec.get("path"),
                model_id=spec.get("model_id"),
                model_label=spec.get("label"),
            )
            self._update_test_qwen_ui()
            _eai_platform_bootstrap.QTimer.singleShot(2500, self._try_switch_chat_to_remote_qwen)
            return

        api = _eai_adapters_models_qwen_local.resolve_local_qwen_viewer_api_base()
        if _eai_adapters_models_qwen_local.check_local_qwen_server_health(api):
            info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api) or {}
            mid = str(info.get("model") or spec.get("label") or key)
            self._append_test_infer_log(
                f"本地 Qwen 已部署，跳过启动: {api} ({mid})"
            )
            if spec.get("path"):
                self._local_qwen_launcher.start(
                    model_dir=spec.get("path"),
                    model_id=spec.get("model_id"),
                    model_label=spec.get("label"),
                )
            else:
                self._local_qwen_launcher.start(model_key=key)
            self._update_test_qwen_ui()
            _eai_platform_bootstrap.QTimer.singleShot(500, self._try_switch_chat_to_local_qwen)
            return

        path = spec.get("path") or _eai_adapters_models_qwen_local.local_qwen_model_dir_for_key(key)
        if spec.get("path"):
            if key.endswith("-a3b") or "35b" in key.lower():
                self._append_test_infer_log(
                    "提示: 本地单卡部署 35B 会走 CPU/磁盘 offload，很慢；"
                    "建议把部署位置改为远程 GPU 机。"
                )
            self._local_qwen_launcher.start(
                model_dir=spec.get("path"),
                model_id=spec.get("model_id"),
                model_label=spec.get("label"),
            )
        else:
            if key == "qwen3.5-35b-a3b":
                self._append_test_infer_log(
                    "提示: 本地单卡部署 35B 会走 CPU/磁盘 offload，很慢；"
                    "建议把部署位置改为远程 GPU 机。"
                )
            self._local_qwen_launcher.start(model_key=key)
        self._update_test_qwen_ui()
        _eai_platform_bootstrap.QTimer.singleShot(2000, self._try_switch_chat_to_local_qwen)

