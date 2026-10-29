"""ui.window.datasets — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_datasets_notes = module_ref('eai_app.ui.datasets.notes')

class DatasetsMixin:
    def _on_dataset_open_survey_file(self) -> None:
        path = _eai_adapters_robot_launch_utils.embodied_open_datasets_notes_path()
        if not _eai_platform_bootstrap.os.path.isfile(path):
            _eai_platform_bootstrap.QMessageBox.warning(self, "数据集", f"文档不存在:\n{path}")
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(_eai_platform_bootstrap.os.path.abspath(path)))

    def _dataset_current_root(self) -> str:
        text = self.dataset_root_edit.text().strip() if hasattr(self, "dataset_root_edit") else ""
        if text:
            return _eai_platform_bootstrap.os.path.abspath(_eai_platform_bootstrap.os.path.expanduser(text))
        return _eai_platform_bootstrap.os.path.abspath(_eai_configuration_chat.EAI_DATASETS_DIR)

    def _dataset_selected_path(self) -> str:
        item = self.dataset_list.currentItem() if hasattr(self, "dataset_list") else None
        if item is None:
            return ""
        return str(item.data(_eai_platform_bootstrap.Qt.UserRole) or "")

    def _on_dataset_preset_changed(self, _index: int = 0) -> None:
        path = str(self.dataset_preset_combo.currentData() or "").strip()
        if not path:
            return
        self.dataset_root_edit.setText(path)
        self._refresh_dataset_browser()

    def _on_dataset_root_browse(self) -> None:
        current = self._dataset_current_root()
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择数据集根目录", current if _eai_platform_bootstrap.os.path.isdir(current) else _eai_configuration_models.EAI_DIR
        )
        if not selected:
            return
        self.dataset_root_edit.setText(selected)
        # 若匹配预设则同步 combo，否则保持当前预设项不动
        for i in range(self.dataset_preset_combo.count()):
            if _eai_platform_bootstrap.os.path.abspath(str(self.dataset_preset_combo.itemData(i) or "")) == _eai_platform_bootstrap.os.path.abspath(
                selected
            ):
                self.dataset_preset_combo.blockSignals(True)
                self.dataset_preset_combo.setCurrentIndex(i)
                self.dataset_preset_combo.blockSignals(False)
                break
        self._refresh_dataset_browser()

    def _on_dataset_refresh_clicked(self) -> None:
        self._refresh_dataset_browser()

    def _on_dataset_up_clicked(self) -> None:
        root = self._dataset_current_root()
        parent = _eai_platform_bootstrap.os.path.dirname(root.rstrip(_eai_platform_bootstrap.os.sep))
        if parent and parent != root and _eai_platform_bootstrap.os.path.isdir(parent):
            self.dataset_root_edit.setText(parent)
            self._refresh_dataset_browser()

    def _on_dataset_filter_changed(self, *_args) -> None:
        self._refresh_dataset_browser(keep_selection=True)

    def _on_dataset_item_activated(self, item: object) -> None:
        if not isinstance(item, _eai_platform_bootstrap.QListWidgetItem):
            return
        path = str(item.data(_eai_platform_bootstrap.Qt.UserRole) or "")
        kind = str(item.data(_eai_platform_bootstrap.Qt.UserRole + 1) or "")
        if kind == "dir" and _eai_platform_bootstrap.os.path.isdir(path):
            self.dataset_root_edit.setText(path)
            self._refresh_dataset_browser()

    def _on_dataset_selection_changed(self) -> None:
        path = self._dataset_selected_path()
        self.dataset_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
        self.dataset_preview.setText("选择左侧条目预览")
        self.dataset_detail_edit.clear()
        if not path:
            return
        kind = "file"
        item = self.dataset_list.currentItem()
        if item is not None:
            kind = str(item.data(_eai_platform_bootstrap.Qt.UserRole + 1) or "file")
        lines: _eai_platform_bootstrap.List[str] = [f"路径: {path}", f"类型: {kind}"]
        try:
            st = _eai_platform_bootstrap.os.stat(path)
            lines.append(f"大小: {_eai_ui_datasets_notes._dataset_format_bytes(st.st_size)}")
            lines.append(
                f"修改: {_eai_platform_bootstrap.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except OSError as exc:
            lines.append(f"stat 失败: {exc}")
            self.dataset_detail_edit.setPlainText("\n".join(lines))
            return
        if kind == "dir":
            try:
                children = [n for n in _eai_platform_bootstrap.os.listdir(path) if not n.startswith(".")]
                lines.append(f"子项: {len(children)}")
                preview_names = sorted(children, key=lambda x: x.lower())[:30]
                if preview_names:
                    lines.append("内容:")
                    lines.extend(f"  - {n}" for n in preview_names)
                    if len(children) > 30:
                        lines.append(f"  … 另有 {len(children) - 30} 项")
            except OSError as exc:
                lines.append(f"列出失败: {exc}")
        elif kind == "image" and _eai_platform_bootstrap.os.path.isfile(path):
            pix = _eai_platform_bootstrap.QPixmap(path)
            if not pix.isNull():
                lines.append(f"分辨率: {pix.width()}×{pix.height()}")
                self.dataset_preview.setPixmap(
                    pix.scaled(
                        self.dataset_preview.size(),
                        _eai_platform_bootstrap.Qt.KeepAspectRatio,
                        _eai_platform_bootstrap.Qt.SmoothTransformation,
                    )
                )
                self.dataset_preview.setText("")
                self.dataset_preview.setToolTip(path)
            else:
                self.dataset_preview.setText("无法加载图像")
        elif kind == "json" and _eai_platform_bootstrap.os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = _eai_platform_bootstrap.json.load(f)
                pretty = _eai_platform_bootstrap.json.dumps(data, ensure_ascii=False, indent=2)
                if len(pretty) > 4000:
                    pretty = pretty[:4000] + "\n…(截断)"
                lines.append("内容预览:")
                lines.append(pretty)
                # Bagel 历史：尝试预览输出图
                if isinstance(data, dict):
                    turns = data.get("turns")
                    if isinstance(turns, list) and turns:
                        last = turns[-1] if isinstance(turns[-1], dict) else {}
                        out_img = str(last.get("output_image") or "")
                        if out_img and _eai_platform_bootstrap.os.path.isfile(out_img):
                            pix = _eai_platform_bootstrap.QPixmap(out_img)
                            if not pix.isNull():
                                self.dataset_preview.setPixmap(
                                    pix.scaled(
                                        self.dataset_preview.size(),
                                        _eai_platform_bootstrap.Qt.KeepAspectRatio,
                                        _eai_platform_bootstrap.Qt.SmoothTransformation,
                                    )
                                )
                                self.dataset_preview.setText("")
                                self.dataset_preview.setToolTip(out_img)
                                lines.append(
                                    f"末轮输出图: {pix.width()}×{pix.height()}"
                                )
            except Exception as exc:
                lines.append(f"JSON 读取失败: {exc}")
        elif kind == "rrd":
            lines.append("RRD 轨迹文件，可点「设为回放」送到回放页。")
        elif kind == "video":
            lines.append("视频文件（可点「打开目录」后用系统播放器打开）。")
        self.dataset_detail_edit.setPlainText("\n".join(lines))

    def _on_dataset_open_dir(self) -> None:
        path = self._dataset_selected_path() or self._dataset_current_root()
        if _eai_platform_bootstrap.os.path.isfile(path):
            path = _eai_platform_bootstrap.os.path.dirname(path)
        if not _eai_platform_bootstrap.os.path.isdir(path):
            _eai_platform_bootstrap.QMessageBox.warning(self, "数据集", f"目录不存在: {path}")
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))

    def _on_dataset_copy_path(self) -> None:
        path = self._dataset_selected_path() or self._dataset_current_root()
        if not path:
            return
        clip = _eai_platform_bootstrap.QApplication.clipboard()
        if clip is not None:
            clip.setText(path)
            self.status_bar.showMessage(f"已复制: {path}", 3000)

    def _on_dataset_to_replay(self) -> None:
        path = self._dataset_selected_path()
        if not path or not path.lower().endswith(".rrd"):
            _eai_platform_bootstrap.QMessageBox.information(self, "数据集", "请先选择一个 .rrd 文件。")
            return
        resolved = _eai_ui_datasets_notes.resolve_rrd_path(path)
        if not _eai_platform_bootstrap.os.path.isfile(resolved):
            _eai_platform_bootstrap.QMessageBox.warning(self, "数据集", f"文件不存在: {resolved}")
            return
        self._selected_rrd_path = resolved
        if hasattr(self, "replay_rrd_path_edit"):
            self.replay_rrd_path_edit.setText(resolved)
        self.status_bar.showMessage(f"已设为回放文件: {resolved}", 4000)

    def _on_dataset_to_bagel(self) -> None:
        path = self._dataset_selected_path()
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            _eai_platform_bootstrap.QMessageBox.information(self, "数据集", "请先选择一个图像文件。")
            return
        ext = _eai_platform_bootstrap.os.path.splitext(path)[1].lower()
        if ext not in _eai_ui_datasets_notes._DATASET_IMAGE_EXTS:
            # Bagel 历史 JSON：取末轮输出图
            if ext == ".json":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = _eai_platform_bootstrap.json.load(f)
                    turns = data.get("turns") if isinstance(data, dict) else None
                    if isinstance(turns, list) and turns:
                        last = turns[-1] if isinstance(turns[-1], dict) else {}
                        cand = str(
                            last.get("output_image") or last.get("input_image") or ""
                        )
                        if cand and _eai_platform_bootstrap.os.path.isfile(cand):
                            path = cand
                        else:
                            raise ValueError("历史中无可用图像")
                    else:
                        raise ValueError("不是含 turns 的 Bagel 历史")
                except Exception as exc:
                    _eai_platform_bootstrap.QMessageBox.information(
                        self, "数据集", f"请选择图像，或含输出图的 Bagel 历史 JSON。\n{exc}"
                    )
                    return
            else:
                _eai_platform_bootstrap.QMessageBox.information(self, "数据集", "请选择图像文件（或 Bagel 历史 JSON）。")
                return
        if not hasattr(self, "_set_bagel_call_input_image"):
            _eai_platform_bootstrap.QMessageBox.warning(self, "数据集", "Bagel 页未就绪。")
            return
        if self._set_bagel_call_input_image(path):
            self.status_bar.showMessage(f"已送入 Bagel 输入: {path}", 4000)
            tabs = getattr(self, "control_tabs", None)
            if tabs is not None:
                for i in range(tabs.count()):
                    if tabs.tabText(i) == "sub image":
                        tabs.setCurrentIndex(i)
                        break
        else:
            _eai_platform_bootstrap.QMessageBox.warning(self, "数据集", f"无法读取图像: {path}")

    def _real_robot_backend_id(self) -> str:
        by_key = getattr(self, "_real_robot_radio_by_key", None) or {}
        for key, radio in by_key.items():
            if radio is not None and radio.isChecked():
                return _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(key))
        if (
            getattr(self, "real_robot_franka_radio", None) is not None
            and self.real_robot_franka_radio.isChecked()
        ):
            return _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA
        if (
            getattr(self, "real_robot_r1_radio", None) is not None
            and self.real_robot_r1_radio.isChecked()
        ):
            return _eai_configuration_policies.REAL_ROBOT_BACKEND_R1
        return _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D

    def _real_robot_backend_radios(self) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.QRadioButton]:
        by_key = getattr(self, "_real_robot_radio_by_key", None) or {}
        if by_key:
            return [r for r in by_key.values() if r is not None]
        radios: _eai_platform_bootstrap.List[_eai_platform_bootstrap.QRadioButton] = []
        for attr in (
            "real_robot_a2d_radio",
            "real_robot_r1_radio",
            "real_robot_franka_radio",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                radios.append(w)
        return radios

    def _real_robot_backend_selector_keys(self) -> _eai_platform_bootstrap.List[str]:
        bar = getattr(self, "real_robot_backend_tab_bar", None)
        if bar is None:
            return []
        out: _eai_platform_bootstrap.List[str] = []
        for i in range(bar.count()):
            key = _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(bar.tabData(i) or ""))
            if key:
                out.append(key)
        return out

    def _on_real_robot_backend_tab_changed(self, index: int) -> None:
        if getattr(self, "_real_robot_tabs_guard", False):
            return
        bar = getattr(self, "real_robot_backend_tab_bar", None)
        if bar is None or index < 0:
            return
        key = _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(bar.tabData(index) or ""))
        radio = (getattr(self, "_real_robot_radio_by_key", None) or {}).get(key)
        if radio is None:
            return
        if radio.isChecked():
            return
        radio.setChecked(True)

    def _on_real_robot_backend_tab_moved(self, _from: int, _to: int) -> None:
        if getattr(self, "_real_robot_tabs_order_guard", False):
            return
        keys = self._real_robot_backend_selector_keys()
        if not keys:
            return
        try:
            path = _eai_infrastructure_ui_storage.save_real_robot_backends_order(keys)
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"已保存真机后端顺序 → {path}", 3000)
        except Exception as exc:
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"保存真机后端顺序失败: {exc}", 4000)

    def _real_robot_backend_has_stack_scripts(self, backend: str = "") -> bool:
        key = _eai_adapters_robot_launch_utils.normalize_real_robot_backend(backend or self._real_robot_backend_id())
        scripts = _eai_adapters_robot_launch_utils.resolve_real_robot_scripts_dir(key)
        return bool(scripts and _eai_platform_bootstrap.os.path.isfile(_eai_platform_bootstrap.os.path.join(scripts, _eai_configuration_policies.ROBOT_SERVICE_SCRIPT)))

    def _on_real_robot_backend_changed(self, *_args) -> None:
        if not hasattr(self, "real_eval_hint_label"):
            return
        # 同步 TabBar 高亮（viewer_ui_state 恢复 radio 时也会走到这里）
        if not getattr(self, "_real_robot_tabs_guard", False):
            self._sync_real_robot_backend_tab_from_radios()
        backend = self._real_robot_backend_id()
        label = _eai_adapters_robot_launch_utils.real_robot_backend_label(backend)
        if backend == _eai_configuration_policies.REAL_ROBOT_BACKEND_R1:
            self.real_eval_hint_label.setText(
                "R1 真机评测：使用 r1-tele / R1_SCRIPTS_DIR（可选 R1_WORKSPACE_DIR）。"
                " 现场 ROS 相机（/camera/*_color），无需 Isaac 桥；"
                "请确认栈已启动、手臂/手已使能，并在「测试」页部署推理服务。"
            )
            self.real_eval_stack_btn.setToolTip(
                "启动 R1 robot stack（R1_SCRIPTS_DIR / r1-tele）"
            )
        elif backend == _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA:
            self.real_eval_hint_label.setText(
                "Franka 真机评测：使用 franka-tele / FRANKA_SCRIPTS_DIR"
                "（可选 FRANKA_WORKSPACE_DIR）。"
                " 现场 ROS 相机（/camera/*_color），无需 Isaac 桥；"
                "请确认栈已启动、手臂/夹爪已使能，并在「测试」页部署推理服务。"
            )
            self.real_eval_stack_btn.setToolTip(
                "启动 Franka robot stack（FRANKA_SCRIPTS_DIR / franka-tele）"
            )
        else:
            self.real_eval_hint_label.setText(
                "A2D 真机评测：使用 a2d-tele / A2D_SCRIPTS_DIR。"
                " 现场 ROS 相机（/camera/*_color），无需 Isaac 桥；"
                "请确认栈已启动、手臂/手已使能，并在「测试」页部署推理服务。"
            )
            self.real_eval_stack_btn.setToolTip(
                "启动 A2D robot stack（与「回放」页相同：A2D_SCRIPTS_DIR / a2d-tele）"
            )
        # 评测进行中禁止切换后端
        self._set_real_robot_backend_switch_enabled(not self._real_eval_active)
        # A2D 采集/标定/HIL 运维面板仅在 A2D 后端显示
        a2d_ops = getattr(self, "a2d_ops_panel", None)
        if a2d_ops is not None:
            a2d_ops.setVisible(backend == _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D)
        has_scripts = self._real_robot_backend_has_stack_scripts(backend)
        if backend in _eai_configuration_policies.REAL_ROBOT_EXTERNAL_STACK_BACKENDS and not has_scripts:
            self.real_eval_stack_btn.setEnabled(True)  # 仍可点，点击后给出配置提示
            self.real_eval_stack_btn.setText(f"启动{label}栈")
        elif not self._real_eval_active:
            self.real_eval_stack_btn.setText(f"启动{label}栈")
        self._refresh_real_eval_status()

    def _on_real_eval_stack_clicked(self) -> None:
        backend = self._real_robot_backend_id()
        label = _eai_adapters_robot_launch_utils.real_robot_backend_label(backend)
        if backend == _eai_configuration_policies.REAL_ROBOT_BACKEND_A2D:
            self._append_real_eval_log(f"[栈] 后端={label}，走 A2D RobotStackLauncher")
            self._on_robot_stack_clicked()
            return
        # R1 / Franka：若已配置同源脚本布局则直接拉起；否则给出配置提示
        scripts = _eai_adapters_robot_launch_utils.resolve_real_robot_scripts_dir(backend)
        robot_script = (
            _eai_platform_bootstrap.os.path.join(scripts, _eai_configuration_policies.ROBOT_SERVICE_SCRIPT) if scripts else ""
        )
        if not scripts or not _eai_platform_bootstrap.os.path.isfile(robot_script):
            hint = _eai_adapters_robot_launch_utils.real_robot_stack_ready_hint(backend)
            env_name = (
                "FRANKA_SCRIPTS_DIR"
                if backend == _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA
                else "R1_SCRIPTS_DIR"
            )
            ws_hint = (
                "/opt/psi/rt/franka-tele（可选 FRANKA_WORKSPACE_DIR）"
                if backend == _eai_configuration_policies.REAL_ROBOT_BACKEND_FRANKA
                else "/opt/psi/rt/r1-tele（可选 R1_WORKSPACE_DIR）"
            )
            msg = (
                f"{label} 真机栈未配置：{hint}\n\n"
                f"请设置环境变量 {env_name} 指向含 "
                f"{_eai_configuration_policies.ROBOT_SERVICE_SCRIPT} 的 scripts 目录，"
                f"或安装 {ws_hint}。"
            )
            self._append_real_eval_log(f"[栈] {hint}")
            self.status_bar.showMessage(f"{label} 真机栈未配置", 6000)
            _eai_platform_bootstrap.QMessageBox.information(self, f"{label} 机器人栈", msg)
            return
        self._append_real_eval_log(
            f"[栈] 后端={label} scripts={scripts}；尝试启动（与 A2D 同构脚本）"
        )
        self._launch_real_robot_stack_detached(backend)

    def _real_eval_start_blockers(self) -> _eai_platform_bootstrap.List[str]:
        """真机「开始评测」不可点时的原因列表。"""
        backend = self._real_robot_backend_id()
        warnings: _eai_platform_bootstrap.List[str] = []
        try:
            names = list(dict(self.node.get_topic_names_and_types()))
            cams = sum(
                1 for t in names if t.startswith("/camera") and t.endswith("_color")
            )
            if cams <= 0:
                warnings.append(
                    "未发现 /camera/*_color topic（确认相机/栈是否在同一 ROS_DOMAIN）"
                )
        except Exception as exc:
            warnings.append(f"topic 查询失败: {exc}")
        if not self.node.is_arm_enabled():
            warnings.append("手臂未使能")
        if not self.node.is_hand_enabled():
            warnings.append("手部未使能")
        if (
            backend in _eai_configuration_policies.REAL_ROBOT_EXTERNAL_STACK_BACKENDS
            and not self._real_robot_backend_has_stack_scripts(backend)
        ):
            warnings.append(_eai_adapters_robot_launch_utils.real_robot_stack_ready_hint(backend))
        return warnings

    def _on_real_eval_start_clicked(self) -> None:
        backend = self._real_robot_backend_id()
        label = _eai_adapters_robot_launch_utils.real_robot_backend_label(backend)
        warnings = self._real_eval_start_blockers()
        if warnings:
            for w in warnings:
                self._append_real_eval_log(f"[就绪检查] {w}")
            self.status_bar.showMessage("真机评测：存在未就绪项，请查看日志", 5000)
            self._refresh_real_eval_status()
            return
        task = self.real_eval_task_edit.text().strip()
        self._append_real_eval_log("[就绪检查] OK")
        self._real_eval_active = True
        self.real_eval_start_btn.setEnabled(False)
        self.real_eval_stop_btn.setEnabled(True)
        self.real_eval_task_edit.setEnabled(False)
        self._set_real_robot_backend_switch_enabled(False)
        if not self._real_eval_timer.isActive():
            self._real_eval_timer.start()
        msg = f"开始真机评测[{label}]" + (f"：{task}" if task else "")
        self._append_real_eval_log(f"--- {msg} ---")
        self.status_bar.showMessage(msg, 4000)
        self._refresh_real_eval_status()

    def _on_real_eval_stop_clicked(self) -> None:
        if not self._real_eval_active:
            return
        backend = self._real_robot_backend_id()
        label = _eai_adapters_robot_launch_utils.real_robot_backend_label(backend)
        self._real_eval_active = False
        self.real_eval_stop_btn.setEnabled(False)
        self.real_eval_task_edit.setEnabled(True)
        self._set_real_robot_backend_switch_enabled(True)
        self._real_eval_timer.stop()
        self._append_real_eval_log(f"--- 真机评测已停止[{label}] ---")
        self.status_bar.showMessage(f"真机评测已停止（{label}）", 3000)
        self._on_real_robot_backend_changed()

    def _on_real_eval_clear_log_clicked(self) -> None:
        self.real_eval_log_edit.clear()

    def _on_real_eval_local_cam_help(self) -> None:
        _eai_platform_bootstrap.QMessageBox.information(
            self, "本机摄像头 → 远端（真机评测）", _eai_configuration_chat.REAL_EVAL_LOCAL_CAM_HELP
        )

    def _on_real_eval_local_cam_toggled(self) -> None:
        if self._real_eval_local_cam_active:
            self._stop_real_eval_local_cam()
            self._append_real_eval_log("[本地摄像头] 已停止预览")
            self.status_bar.showMessage("本地摄像头预览已停止", 3000)
            return
        url = self.real_eval_cam_url_edit.text().strip() or _eai_configuration_chat.SKELETON_NETWORK_URL_DEFAULT
        self.real_eval_cam_url_edit.setText(url)
        self.real_eval_cam_status_label.setText(f"本地摄像头: 连接中… {url}")
        self.real_eval_cam_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        self.status_bar.showMessage(f"正在连接本地摄像头流: {url}")
        _eai_platform_bootstrap.QApplication.processEvents()
        ok, msg = self._open_real_eval_local_cam_source(url)
        if not ok:
            # 远端无流时，若存在 /dev/video0（如 v4l2loopback）则尝试本机
            fallback = None
            if _eai_platform_bootstrap.os.path.exists("/dev/video0"):
                fallback = "0"
            if fallback is not None and url != fallback:
                self._append_real_eval_log(f"[本地摄像头] 流失败，尝试 /dev/video0 …")
                ok2, msg2 = self._open_real_eval_local_cam_source(fallback)
                if ok2:
                    ok, msg = ok2, msg2
                else:
                    _eai_platform_bootstrap.QMessageBox.warning(self, "本地摄像头预览", f"{msg}\n\n本机回退也失败:\n{msg2}")
                    self.real_eval_cam_status_label.setText("本地摄像头: 连接失败")
                    self.real_eval_cam_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
                    return
            else:
                _eai_platform_bootstrap.QMessageBox.warning(self, "本地摄像头预览", msg)
                self.real_eval_cam_status_label.setText("本地摄像头: 连接失败")
                self.real_eval_cam_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
                return
        self._real_eval_local_cam_active = True
        self.real_eval_cam_url_edit.setEnabled(False)
        self.real_eval_cam_preview_btn.setText("停止预览")
        short = msg if len(msg) <= 56 else msg[:53] + "..."
        self.real_eval_cam_status_label.setText(f"本地摄像头: 预览中 @ {short}")
        self.real_eval_cam_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        self.real_eval_cam_preview_label.setText("")
        self._real_eval_local_cam_timer.start()
        self._append_real_eval_log(f"[本地摄像头] 预览已开启: {msg}")
        self.status_bar.showMessage("本地摄像头预览已开启", 3000)

    def _on_real_eval_local_cam_tick(self) -> None:
        if not self._real_eval_local_cam_active or self._real_eval_local_cam_busy:
            return
        cap = self._real_eval_local_cap
        if cap is None:
            return
        self._real_eval_local_cam_busy = True
        try:
            ok, frame = cap.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                self.real_eval_cam_status_label.setText("本地摄像头: 等待画面…")
                return
            max_w = max(320, self.real_eval_cam_preview_label.width() - 8)
            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / float(w)
                frame = _eai_platform_bootstrap.cv2.resize(
                    frame,
                    (max_w, max(1, int(h * scale))),
                    interpolation=_eai_platform_bootstrap.cv2.INTER_AREA,
                )
            self.real_eval_cam_preview_label.setPixmap(_eai_adapters_robot_geometry.cv2_to_qpixmap(frame))
        except Exception as exc:
            self.real_eval_cam_status_label.setText(f"本地摄像头: 读帧错误 {exc}")
            self.real_eval_cam_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
        finally:
            self._real_eval_local_cam_busy = False

    def _on_replay_select_clicked(self) -> None:
        if self._replay_launcher.is_running():
            return

        initial_dir = _eai_adapters_robot_launch_utils.default_rrd_dataset_dir()
        if self._selected_rrd_path and _eai_platform_bootstrap.os.path.isfile(self._selected_rrd_path):
            initial_dir = _eai_platform_bootstrap.os.path.dirname(self._selected_rrd_path)
        elif not _eai_platform_bootstrap.os.path.isdir(initial_dir):
            initial_dir = _eai_platform_bootstrap.os.path.expanduser("~")
        rrd_path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 RRD 回放文件",
            initial_dir,
            "RRD Files (*.rrd);;All Files (*)",
        )
        if not rrd_path:
            return

        resolved_path = _eai_ui_datasets_notes.resolve_rrd_path(rrd_path)
        self._selected_rrd_path = resolved_path
        self._update_replay_rrd_path_display(resolved_path)
        self._update_replay_ui()

    def _on_replay_start_clicked(self) -> None:
        if self._replay_launcher.is_running():
            return

        path = (self._selected_rrd_path or "").strip()
        if not path:
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "RRD 回放",
                "请先点击「选择路径」选择 .rrd 文件。",
            )
            return
        if not _eai_platform_bootstrap.os.path.isfile(path):
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "RRD 回放",
                f"文件不存在：\n{path}",
            )
            return
        self._start_rrd_replay(path, self.replay_count_spin.value())

    def _on_replay_stop_clicked(self) -> None:
        if self._replay_launcher.is_running():
            self._replay_launcher.stop()
        self._update_ctx_ui()

