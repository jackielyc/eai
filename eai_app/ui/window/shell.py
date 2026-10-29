"""ui.window.shell — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_bagel_runtime = module_ref('eai_app.adapters.models.bagel_runtime')
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_datasets_notes = module_ref('eai_app.ui.datasets.notes')

class ShellMixin:
    def _get_psi_policy_dir(self) -> _eai_platform_bootstrap.Optional[str]:
        if self._psi_policy_dir_override:
            return _eai_perception_segmentation.resolve_psi_policy_dir(self._psi_policy_dir_override)
        return _eai_perception_segmentation.resolve_psi_policy_dir()

    def _update_train_path_ui(self) -> None:
        repo = self._get_psi_policy_dir()
        if repo:
            self.train_policy_dir_edit.setText(repo)
            self.train_policy_dir_edit.setToolTip(repo)
        else:
            self.train_policy_dir_edit.clear()
            self.train_policy_dir_edit.setToolTip("未找到 psi-policy 仓库")
        self.train_status_label.setToolTip(
            f"psi-policy 目录: {repo or '(未设置)'}\n训练日志实时显示在下方"
        )
        running = self._train_launcher.is_running()
        self.train_start_btn.setEnabled(not running and repo is not None)
        self.train_browse_policy_btn.setEnabled(not running)

    def _refresh_train_config_combo(self) -> None:
        repo = self._get_psi_policy_dir()
        current = str(self.train_config_combo.currentData() or _eai_configuration_policies.PSI_POLICY_CONFIG_DEFAULT)
        self.train_config_combo.blockSignals(True)
        self.train_config_combo.clear()
        configs = _eai_perception_segmentation.list_psi_policy_workspace_configs(repo) if repo else []
        if not configs:
            configs = [_eai_configuration_policies.PSI_POLICY_CONFIG_DEFAULT]
        for cfg_name in configs:
            self.train_config_combo.addItem(cfg_name, cfg_name)
        idx = self.train_config_combo.findData(current)
        if idx < 0:
            idx = self.train_config_combo.findData(_eai_configuration_policies.PSI_POLICY_CONFIG_DEFAULT)
        if idx >= 0:
            self.train_config_combo.setCurrentIndex(idx)
        self.train_config_combo.blockSignals(False)

    def _append_train_log(self, line: str) -> None:
        self.train_log_edit.append(line)
        scrollbar = self.train_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_train_ui(self, *_args) -> None:
        running = self._train_launcher.is_running()
        self.train_stop_btn.setEnabled(running)
        if running:
            self.train_status_label.setText("训练: 运行中")
            self.train_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.train_status_label.setText("训练: 空闲")
            self.train_status_label.setStyleSheet("")
        self._update_train_path_ui()

    def _append_bagel_log(self, line: str) -> None:
        if _eai_adapters_models_bagel_runtime._is_noisy_bagel_log_line(line):
            return
        self.bagel_log_edit.append(line)
        scrollbar = self.bagel_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _sync_bagel_gradio_embed(self) -> None:
        """Gradio 未启动时收起嵌入页，把高度让给上方输入/输出图。"""
        running = False
        launcher = getattr(self, "_bagel_launcher", None)
        if launcher is not None:
            running = bool(launcher.is_running())
        view = getattr(self, "bagel_web_view", None)
        container = getattr(self, "_bagel_web_container", None)
        embed = container if container is not None else view
        fallback = getattr(self, "_bagel_web_fallback", None)
        preview = getattr(self, "bagel_call_output_preview", None)
        tab = getattr(self, "bagel_tab", None)
        outer = tab.layout() if tab is not None else None
        qmax = 16777215

        def _set_item_stretch(target, stretch: int) -> None:
            if outer is None or target is None:
                return
            for i in range(outer.count()):
                item = outer.itemAt(i)
                if item is None:
                    continue
                if item.widget() is target or item.layout() is target:
                    outer.setStretch(i, stretch)
                    return

        if running:
            if embed is not None:
                embed.setMinimumHeight(280)
                embed.setMaximumHeight(qmax)
                embed.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
                embed.show()
                _set_item_stretch(embed, 1)
            if view is not None:
                view.setMinimumHeight(280)
                view.setMaximumHeight(qmax)
                view.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
                view.show()
            if fallback is not None:
                fallback.setMinimumHeight(80)
                fallback.show()
                _set_item_stretch(fallback, 1)
            if preview is not None:
                preview.setMinimumHeight(90)
                preview.setMaximumHeight(160)
                preview.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Preferred)
            _set_item_stretch(getattr(self, "_bagel_preview_layout", None), 0)
        else:
            if view is not None:
                view.hide()
                view.setMinimumHeight(0)
                view.setMaximumHeight(0)
            if embed is not None:
                embed.hide()
                embed.setMinimumHeight(0)
                embed.setMaximumHeight(0)
                _set_item_stretch(embed, 0)
            if fallback is not None:
                fallback.hide()
                fallback.setMinimumHeight(0)
                _set_item_stretch(fallback, 0)
            if preview is not None:
                preview.setMinimumHeight(160)
                preview.setMaximumHeight(qmax)
                preview.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
            _set_item_stretch(getattr(self, "_bagel_preview_layout", None), 1)
        if tab is not None:
            tab.updateGeometry()

    def _update_bagel_run_ui(self, *_args) -> None:
        running = self._bagel_launcher.is_running()
        adopted = bool(getattr(self._bagel_launcher, "_adopted", False))
        ready = running or bool(self._bagel_launcher.current_url())
        # 复用中仍允许再点启动（会再次探测）；仅本窗口自启进程时禁用
        self.bagel_start_btn.setEnabled(not running or adopted)
        self.bagel_stop_btn.setEnabled(running)
        self.bagel_open_btn.setEnabled(ready)
        if hasattr(self, "bagel_pick_image_btn"):
            self.bagel_pick_image_btn.setEnabled(
                ready and self.bagel_web_view is not None
            )
        for w in (
            self.bagel_root_edit,
            self.bagel_root_browse_btn,
            self.bagel_python_edit,
            self.bagel_python_browse_btn,
            self.bagel_model_edit,
            self.bagel_model_browse_btn,
            self.bagel_host_edit,
            self.bagel_port_spin,
            self.bagel_mode_combo,
            self.bagel_zh_check,
            self.bagel_share_check,
        ):
            w.setEnabled(not running or adopted)
        if running:
            self.bagel_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            if not self._bagel_ready_poll_timer.isActive():
                self._bagel_ready_poll_timer.start()
        else:
            self.bagel_status_label.setStyleSheet("")
            self._bagel_ready_poll_timer.stop()
            if self.bagel_web_view is not None and not running:
                self.bagel_web_view.setHtml(
                    "<html><body style='margin:0;background:#1a1a1a;color:#888;"
                    "font-family:sans-serif;display:flex;align-items:center;"
                    "justify-content:center;height:100vh;'>"
                    "Bagel Gradio 未启动</body></html>"
                )
        self._sync_bagel_gradio_embed()

    def _adopt_persisted_bagel(self) -> None:
        """界面重新打开时，按端口上的真实服务刷新 Bagel 状态。"""
        if not hasattr(self, "_bagel_launcher"):
            return
        host = _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        if hasattr(self, "bagel_host_edit"):
            host = self.bagel_host_edit.text().strip() or host
        api_port = _eai_configuration_models.BAGEL_API_PORT_DEFAULT
        if hasattr(self, "bagel_api_port_spin"):
            api_port = int(self.bagel_api_port_spin.value())
        gradio_port = _eai_configuration_models.BAGEL_SERVER_PORT_DEFAULT
        if hasattr(self, "bagel_port_spin"):
            gradio_port = int(self.bagel_port_spin.value())
        mode = 1
        if hasattr(self, "bagel_mode_combo"):
            mode = int(self.bagel_mode_combo.currentData() or 1)
        if getattr(self, "_bagel_api_launcher", None) is not None:
            if self._bagel_api_launcher.try_adopt(server_name=host, server_port=api_port):
                self._update_bagel_api_ui()
        if self._bagel_launcher.try_adopt(
            server_name=host, server_port=gradio_port, mode=mode
        ):
            self._update_bagel_run_ui()

    def _want_control_tab(self, title: str) -> bool:
        """--tab 过滤时是否需要构建/预热该控制区页。"""
        only = getattr(self, "_only_tabs", None) or []
        if not only:
            return True
        return title in only

    def _ensure_bagel_web_view(self):
        """按需创建 Bagel QWebEngineView（避免启动期 OpenGL/SIP 崩溃）。"""
        if getattr(self, "bagel_web_view", None) is not None:
            return self.bagel_web_view
        if _eai_platform_bootstrap.QWebEngineView is None:
            return None
        if not self._want_control_tab("sub image"):
            return None
        container = getattr(self, "_bagel_web_container", None)
        layout = getattr(self, "_bagel_web_container_layout", None)
        if container is None or layout is None:
            return None
        try:
            view = _eai_platform_bootstrap.QWebEngineView(container)
            view.setMinimumHeight(0)
            view.setStyleSheet(f"QWebEngineView {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; }}")
            try:
                view.page().setBackgroundColor(_eai_platform_bootstrap.QColor("#1a1a1a"))
            except Exception:
                pass
            if _eai_adapters_models_bagel_runtime.BagelWebEnginePage is not None:
                page = _eai_adapters_models_bagel_runtime.BagelWebEnginePage(view)
                view.setPage(page)
                self.bagel_web_page = page
                try:
                    page.setBackgroundColor(_eai_platform_bootstrap.QColor("#1a1a1a"))
                except Exception:
                    pass
                try:
                    settings = view.settings()
                    if _eai_platform_bootstrap.QWebEngineSettings is not None:
                        settings.setAttribute(
                            _eai_platform_bootstrap.QWebEngineSettings.LocalContentCanAccessRemoteUrls, True
                        )
                        settings.setAttribute(
                            _eai_platform_bootstrap.QWebEngineSettings.LocalContentCanAccessFileUrls, True
                        )
                        settings.setAttribute(
                            _eai_platform_bootstrap.QWebEngineSettings.JavascriptEnabled, True
                        )
                        settings.setAttribute(
                            _eai_platform_bootstrap.QWebEngineSettings.LocalStorageEnabled, True
                        )
                except Exception:
                    pass
            else:
                self.bagel_web_page = None
            placeholder = getattr(self, "_bagel_web_placeholder", None)
            if placeholder is not None:
                layout.removeWidget(placeholder)
                placeholder.hide()
                placeholder.deleteLater()
                self._bagel_web_placeholder = None
            layout.addWidget(view, 0)
            self.bagel_web_view = view
            container.show()
            return view
        except Exception as exc:
            print(f"[warn] Bagel WebEngine 创建失败: {exc}", flush=True)
            return None

    def _ensure_astra_web_view(self):
        """按需创建 Astra QWebEngineView（避免启动期 OpenGL/SIP 崩溃）。"""
        if getattr(self, "astra_web_view", None) is not None:
            return self.astra_web_view
        if _eai_platform_bootstrap.QWebEngineView is None:
            return None
        if not self._want_control_tab("Astra"):
            return None
        container = getattr(self, "_astra_web_container", None)
        layout = getattr(self, "_astra_web_container_layout", None)
        if container is None or layout is None:
            return None
        try:
            view = _eai_platform_bootstrap.QWebEngineView(container)
            view.setMinimumHeight(360)
            view.setStyleSheet("QWebEngineView { background-color: #111111; }")
            try:
                view.page().setBackgroundColor(_eai_platform_bootstrap.QColor("#111111"))
            except Exception:
                pass
            try:
                settings = view.settings()
                if _eai_platform_bootstrap.QWebEngineSettings is not None:
                    settings.setAttribute(_eai_platform_bootstrap.QWebEngineSettings.JavascriptEnabled, True)
                    settings.setAttribute(_eai_platform_bootstrap.QWebEngineSettings.LocalStorageEnabled, True)
                    settings.setAttribute(_eai_platform_bootstrap.QWebEngineSettings.PluginsEnabled, True)
            except Exception:
                pass
            placeholder = getattr(self, "_astra_web_placeholder", None)
            if placeholder is not None:
                layout.removeWidget(placeholder)
                placeholder.hide()
                placeholder.deleteLater()
                self._astra_web_placeholder = None
            layout.addWidget(view, 1)
            self.astra_web_view = view
            url = self._astra_current_url()
            view.setUrl(_eai_platform_bootstrap.QUrl(url))
            return view
        except Exception as exc:
            print(f"[warn] Astra WebEngine 创建失败: {exc}", flush=True)
            return None

    def _astra_current_url(self) -> str:

        if hasattr(self, "astra_url_edit"):
            text = self.astra_url_edit.text().strip()
            if text:
                return text
        return _eai_configuration_models.ASTRA_ROBODOJO_URL

    def _load_astra_url(self, url: str, *, sync_combo: bool = True) -> None:
        url = (url or "").strip() or _eai_configuration_models.ASTRA_ROBODOJO_URL
        if hasattr(self, "astra_url_edit"):
            self.astra_url_edit.setText(url)
        if sync_combo and hasattr(self, "astra_page_combo"):
            matched = -1
            for i in range(self.astra_page_combo.count()):
                preset = str(self.astra_page_combo.itemData(i) or "")
                if preset.rstrip("/") == url.rstrip("/") or preset == url:
                    matched = i
                    break
            self.astra_page_combo.blockSignals(True)
            if matched >= 0:
                self.astra_page_combo.setCurrentIndex(matched)
            self.astra_page_combo.blockSignals(False)
        view = self._ensure_astra_web_view()
        if view is not None:
            view.setUrl(_eai_platform_bootstrap.QUrl(url))
        tip = f"Astra: {url}"
        if _eai_adapters_models_discovery.astra_url_prefers_external_browser(url):
            tip += "（视频播不出时点「浏览器打开」）"
        self.status_bar.showMessage(tip)

    def _on_astra_page_changed(self, _index: int = 0) -> None:
        if not hasattr(self, "astra_page_combo"):
            return
        url = str(self.astra_page_combo.currentData() or "").strip()
        if url:
            self._load_astra_url(url, sync_combo=False)

    def _on_astra_go_clicked(self) -> None:
        self._load_astra_url(self._astra_current_url())

    def _on_astra_reload_clicked(self) -> None:
        view = self._ensure_astra_web_view()
        if view is None:
            return
        url = self._astra_current_url()
        view.setUrl(_eai_platform_bootstrap.QUrl(url))
        self.status_bar.showMessage(f"已刷新 {url}")

    def _on_astra_open_clicked(self) -> None:
        url = self._astra_current_url()
        ok, detail = _eai_ui_datasets_notes.open_url_in_chromium_app(url)
        if ok:
            self.status_bar.showMessage(f"已用 {detail} 打开 {url}")
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl(url))
        self.status_bar.showMessage(f"已打开 {url}（{detail}）")

    def _append_humanego_log(self, line: str) -> None:
        self.humanego_log_edit.append(line)
        bar = self.humanego_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_humanego_status(self, text: str) -> None:
        msg = text if text.startswith("HumanEgo") else f"HumanEgo: {text}"
        self.humanego_status_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _update_humanego_ui(self, *_args) -> None:
        running = self._humanego_launcher.is_running()
        self.humanego_run_btn.setEnabled(not running)
        self.humanego_stop_btn.setEnabled(running)
        for w in (
            self.humanego_root_edit,
            self.humanego_root_browse_btn,
            self.humanego_python_edit,
            self.humanego_python_browse_btn,
            self.humanego_action_combo,
            self.humanego_task_combo,
            self.humanego_num_edit,
            self.humanego_input_only_check,
            self.humanego_use_cfg_check,
            self.humanego_mps_edit,
            self.humanego_mps_browse_btn,
            self.humanego_mps_combo,
            self.humanego_cmd_edit,
        ):
            w.setEnabled(not running)
        if running:
            self.humanego_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.humanego_status_label.setStyleSheet("")

    def _on_humanego_root_browse(self) -> None:
        cur = self.humanego_root_edit.text().strip() or _eai_configuration_models.HUMANOGO_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 HumanEgo 仓库", cur)
        if selected:
            self.humanego_root_edit.setText(selected)
            self.humanego_python_edit.setText(_eai_configuration_tabs.resolve_humanego_python(selected))
            self._on_humanego_refresh_mps()
            self._on_humanego_action_changed()

    def _on_humanego_python_browse(self) -> None:
        cur = self.humanego_python_edit.text().strip() or _eai_platform_bootstrap.sys.executable
        initial = _eai_platform_bootstrap.os.path.dirname(cur) if cur else ""
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self, "选择 Python", initial, "Python (python*);;All (*)"
        )
        if selected:
            self.humanego_python_edit.setText(selected)

    def _on_humanego_mps_browse(self) -> None:
        repo = _eai_configuration_tabs.resolve_humanego_root(self.humanego_root_edit.text())
        cur = self.humanego_mps_edit.text().strip() or _eai_platform_bootstrap.os.path.join(repo, "data")
        if not _eai_platform_bootstrap.os.path.isdir(cur):
            cur = repo
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 mps_* recording 目录", cur)
        if selected:
            self.humanego_mps_edit.setText(selected)

    def _on_humanego_refresh_mps(self) -> None:
        repo = _eai_configuration_tabs.resolve_humanego_root(self.humanego_root_edit.text())
        found: _eai_platform_bootstrap.List[str] = []
        data_root = _eai_platform_bootstrap.os.path.join(repo, "data")
        if _eai_platform_bootstrap.os.path.isdir(data_root):
            for task in sorted(_eai_platform_bootstrap.os.listdir(data_root)):
                aria = _eai_platform_bootstrap.os.path.join(data_root, task, "aria")
                if not _eai_platform_bootstrap.os.path.isdir(aria):
                    continue
                for name in sorted(_eai_platform_bootstrap.os.listdir(aria)):
                    path = _eai_platform_bootstrap.os.path.join(aria, name)
                    if _eai_platform_bootstrap.os.path.isdir(path) and name.startswith("mps_"):
                        found.append(path)
        self.humanego_mps_combo.blockSignals(True)
        self.humanego_mps_combo.clear()
        self.humanego_mps_combo.addItem("(未选择)", "")
        for path in found:
            rel = path
            try:
                rel = _eai_platform_bootstrap.os.path.relpath(path, repo)
            except ValueError:
                pass
            self.humanego_mps_combo.addItem(rel, path)
        self.humanego_mps_combo.blockSignals(False)
        if found and not self.humanego_mps_edit.text().strip():
            self.humanego_mps_edit.setText(found[0])
            self.humanego_mps_combo.setCurrentIndex(1)

    def _on_humanego_mps_combo_changed(self, _index: int = 0) -> None:
        path = str(self.humanego_mps_combo.currentData() or "").strip()
        if path:
            self.humanego_mps_edit.setText(path)

    def _humanego_build_command(self) -> _eai_platform_bootstrap.Tuple[str, str]:
        action = str(self.humanego_action_combo.currentData() or "download")
        task = str(self.humanego_task_combo.currentData() or "serve_bread")
        py = "$PYTHON"
        if action == "custom":
            cmd = self.humanego_cmd_edit.text().strip()
            return cmd, "custom"
        if action == "setup":
            return "bash setup.sh", "setup.sh"
        if action == "download":
            num = self.humanego_num_edit.text().strip() or "2"
            parts = [
                f"{py} scripts/download_data.py",
                f"--task {_eai_platform_bootstrap.shlex.quote(task)}",
                f"--num {_eai_platform_bootstrap.shlex.quote(num)}",
            ]
            if self.humanego_input_only_check.isChecked():
                parts.append("--input-only")
            return " ".join(parts), f"download {task}"
        if action == "preprocess":
            mps = self.humanego_mps_edit.text().strip()
            if not mps:
                return "", "preprocess"
            # task all 不适合 preprocess；用 serve_bread 默认
            pp_task = task if task != "all" else "serve_bread"
            return (
                f"{py} -m preprocess.Preprocess "
                f"--mps_path {_eai_platform_bootstrap.shlex.quote(mps)} --task {_eai_platform_bootstrap.shlex.quote(pp_task)}",
                f"preprocess {_eai_platform_bootstrap.os.path.basename(mps)}",
            )
        if action == "train":
            tr_task = task if task != "all" else "serve_bread"
            parts = [
                f"{py} -m training.FlowMatchingTrainer",
                f"--task {_eai_platform_bootstrap.shlex.quote(tr_task)}",
                "--job HumanEgo",
            ]
            if self.humanego_use_cfg_check.isChecked():
                parts.append("--use_cfg")
            return " ".join(parts), f"train {tr_task}"
        return self.humanego_cmd_edit.text().strip(), action

    def _on_humanego_action_changed(self, *_args) -> None:
        action = str(self.humanego_action_combo.currentData() or "download")
        need_mps = action == "preprocess"
        self.humanego_mps_edit.setEnabled(True)
        self.humanego_mps_browse_btn.setEnabled(True)
        self.humanego_num_edit.setEnabled(action == "download")
        self.humanego_input_only_check.setEnabled(action == "download")
        self.humanego_use_cfg_check.setEnabled(action == "train")
        self.humanego_task_combo.setEnabled(action in ("download", "preprocess", "train"))
        if action == "custom":
            return
        cmd, _label = self._humanego_build_command()
        if action == "preprocess" and not self.humanego_mps_edit.text().strip():
            cmd = (
                "$PYTHON -m preprocess.Preprocess "
                "--mps_path <mps_path> --task serve_bread"
            )
        if need_mps and not self.humanego_mps_edit.text().strip():
            pass
        self.humanego_cmd_edit.setText(cmd)

    def _on_humanego_run_clicked(self) -> None:
        if self.humanego_action_combo.currentData() != "custom":
            self._on_humanego_action_changed()
        cmd, label = self._humanego_build_command()
        if not cmd.strip():
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "HumanEgo",
                "命令为空。预处理请先填写 mps_path（可点「扫描 data/」）。",
            )
            return
        action = str(self.humanego_action_combo.currentData() or "")
        if action == "preprocess":
            mps = self.humanego_mps_edit.text().strip()
            if not mps or not _eai_platform_bootstrap.os.path.isdir(mps):
                _eai_platform_bootstrap.QMessageBox.warning(self, "HumanEgo", f"mps_path 不存在: {mps}")
                return
        self._humanego_launcher.start_command(
            repo_dir=self.humanego_root_edit.text(),
            python_bin=self.humanego_python_edit.text(),
            command=cmd,
            label=label,
        )
        self._update_humanego_ui()

    def _on_humanego_stop_clicked(self) -> None:
        self._humanego_launcher.stop()
        self._update_humanego_ui()

    def _on_humanego_open_dir(self) -> None:
        repo = _eai_configuration_tabs.resolve_humanego_root(self.humanego_root_edit.text())
        if not _eai_platform_bootstrap.os.path.isdir(repo):
            _eai_platform_bootstrap.QMessageBox.warning(self, "HumanEgo", f"目录不存在: {repo}")
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(repo))

    def _refresh_dataset_open_survey(self) -> None:
        """加载并渲染 docs/embodied-open-datasets-notes.md 到数据集 → 开源梳理。"""
        path = _eai_adapters_robot_launch_utils.embodied_open_datasets_notes_path()
        if hasattr(self, "dataset_survey_path_label"):
            self.dataset_survey_path_label.setText(path)
        view = getattr(self, "dataset_survey_view", None)
        if view is None:
            return
        if not _eai_platform_bootstrap.os.path.isfile(path):
            view.setHtml(
                "<html><body style='background:#1a1b1e;color:#ccc;padding:12px;'>"
                f"<p>未找到分析文档：</p><pre>{_eai_adapters_robot_launch_utils._html_escape_text(path)}</pre>"
                "<p>请确认 <code>docs/embodied-open-datasets-notes.md</code> 存在。</p>"
                "</body></html>"
            )
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(f"开源梳理文档缺失: {path}", 5000)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                md = f.read()
            view.setHtml(_eai_ui_datasets_notes.markdown_notes_to_html(md))
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(f"已加载开源梳理: {path}", 3000)
        except Exception as exc:
            view.setPlainText(f"读取失败: {path}\n{exc}")
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage(f"开源梳理加载失败: {exc}", 5000)

    def _refresh_dataset_browser(self, *, keep_selection: bool = False) -> None:
        if not hasattr(self, "dataset_list"):
            return
        prev = self._dataset_selected_path() if keep_selection else ""
        root = self._dataset_current_root()
        self._dataset_cwd = root
        self.dataset_list.clear()
        self.dataset_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
        self.dataset_preview.setText("选择左侧条目预览")
        self.dataset_detail_edit.clear()
        if not _eai_platform_bootstrap.os.path.isdir(root):
            self.dataset_detail_edit.setPlainText(f"目录不存在:\n{root}")
            if getattr(self, "status_bar", None) is not None:
                self.status_bar.showMessage(f"数据集目录不存在: {root}", 4000)
            return
        filter_text = self.dataset_filter_edit.text().strip().lower()
        type_key = str(self.dataset_type_combo.currentData() or "all")
        try:
            names = sorted(_eai_platform_bootstrap.os.listdir(root), key=lambda x: x.lower())
        except OSError as exc:
            self.dataset_detail_edit.setPlainText(f"无法读取目录: {exc}")
            return
        restore_row = -1
        for name in names:
            if name.startswith("."):
                continue
            if filter_text and filter_text not in name.lower():
                continue
            path = _eai_platform_bootstrap.os.path.join(root, name)
            is_dir = _eai_platform_bootstrap.os.path.isdir(path)
            ext = _eai_platform_bootstrap.os.path.splitext(name)[1].lower()
            kind = "dir" if is_dir else "file"
            if is_dir:
                kind = "dir"
            elif ext in _eai_ui_datasets_notes._DATASET_IMAGE_EXTS:
                kind = "image"
            elif ext in _eai_ui_datasets_notes._DATASET_VIDEO_EXTS:
                kind = "video"
            elif ext in _eai_ui_datasets_notes._DATASET_RRD_EXTS:
                kind = "rrd"
            elif ext in _eai_ui_datasets_notes._DATASET_JSON_EXTS:
                kind = "json"
            if type_key != "all" and kind != type_key:
                continue
            try:
                st = _eai_platform_bootstrap.os.stat(path)
                size_txt = "DIR" if is_dir else _eai_ui_datasets_notes._dataset_format_bytes(st.st_size)
                mtime = _eai_platform_bootstrap.datetime.fromtimestamp(st.st_mtime).strftime("%m-%d %H:%M")
            except OSError:
                size_txt = "?"
                mtime = ""
            prefix = {
                "dir": "[D]",
                "image": "[I]",
                "video": "[V]",
                "rrd": "[R]",
                "json": "[J]",
            }.get(kind, "[F]")
            label = f"{prefix} {name}  ·  {size_txt}"
            if mtime:
                label += f"  ·  {mtime}"
            item = _eai_platform_bootstrap.QListWidgetItem(label)
            item.setData(_eai_platform_bootstrap.Qt.UserRole, path)
            item.setData(_eai_platform_bootstrap.Qt.UserRole + 1, kind)
            item.setToolTip(path)
            self.dataset_list.addItem(item)
            if prev and _eai_platform_bootstrap.os.path.abspath(path) == _eai_platform_bootstrap.os.path.abspath(prev):
                restore_row = self.dataset_list.count() - 1
        if restore_row >= 0:
            self.dataset_list.setCurrentRow(restore_row)
        if getattr(self, "status_bar", None) is not None:
            self.status_bar.showMessage(
                f"数据集: {self.dataset_list.count()} 项 @ {root}", 4000
            )

    def _poll_bagel_http_ready(self) -> None:
        if not self._bagel_launcher.is_running():
            self._bagel_ready_poll_timer.stop()
            return
        url = self._bagel_launcher.current_url()
        if not url:
            return
        try:
            req = _eai_platform_bootstrap.urllib.request.Request(url, method="GET")
            with _eai_platform_bootstrap.urllib.request.urlopen(req, timeout=1.5) as resp:
                if 200 <= int(getattr(resp, "status", 200)) < 500:
                    self._bagel_ready_poll_timer.stop()
                    self._load_bagel_web(url)
                    self.bagel_status_label.setText(
                        f"Bagel: 已就绪 {url}  ·  mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self._bagel_launcher.current_mode())}"
                    )
                    self.bagel_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
                    self.bagel_open_btn.setEnabled(True)
                    if hasattr(self, "bagel_pick_image_btn"):
                        self.bagel_pick_image_btn.setEnabled(
                            self.bagel_web_view is not None
                        )
                    self._append_bagel_log(
                        f"[ready] {url}  mode={_eai_adapters_models_bagel_runtime.bagel_mode_label(self._bagel_launcher.current_mode())} — 用「选择图片」上传到嵌入页"
                    )
        except Exception:
            pass

    def _update_bagel_api_ui(self, *_args) -> None:
        running = self._bagel_api_launcher.is_running()
        ready = bool(self._bagel_api_launcher._ready_emitted) and running
        adopted = bool(getattr(self._bagel_api_launcher, "_adopted", False))
        # 复用中仍允许再点部署（会再次探测）；仅本窗口自启进程时禁用
        self.bagel_api_start_btn.setEnabled(not running or adopted)
        self.bagel_api_stop_btn.setEnabled(running)
        self.bagel_api_port_spin.setEnabled(not running or adopted)
        self.bagel_call_btn.setEnabled(running and ready and not self._bagel_call_busy)
        if running:
            self.bagel_api_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            if not self._bagel_api_ready_poll_timer.isActive() and not ready:
                self._bagel_api_ready_poll_timer.start()
            if adopted and ready:
                self._bagel_api_ready_poll_timer.stop()
        else:
            self.bagel_api_status_label.setStyleSheet("")
            self._bagel_api_ready_poll_timer.stop()
            if not ready:
                self.bagel_api_status_label.setText("推理 API: 未部署")
            try:
                self.chat_panel.set_extra_inference_candidates([])
                self.chat_panel.refresh_running_providers(notify=False, silent_apply=True)
            except Exception:
                pass

    def _poll_bagel_api_ready(self) -> None:
        if not self._bagel_api_launcher.is_running():
            self._bagel_api_ready_poll_timer.stop()
            return
        host = getattr(self._bagel_api_launcher, "_adopt_host", None) or (
            self.bagel_host_edit.text().strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        )
        port = int(
            getattr(self._bagel_api_launcher, "_adopt_port", 0)
            or self.bagel_api_port_spin.value()
        )
        # 必须确认是 Bagel serve_api，避免其它服务 /health 误判就绪
        body = _eai_adapters_models_bagel_runtime.bagel_probe_api_health(host, port, timeout_s=1.5)
        if body is None:
            return
        url = self._bagel_api_launcher.current_url().rstrip("/")
        if not url:
            url = _eai_adapters_models_bagel_runtime.bagel_app_url(host, port).rstrip("/")
        self._bagel_api_ready_poll_timer.stop()
        if not self._bagel_api_launcher._ready_emitted:
            self._bagel_api_launcher._ready_emitted = True
            self._bagel_api_launcher.ready_url.emit(url + "/")

    def _set_bagel_call_input_image(self, path: str, display_name: str = "") -> bool:
        path = str(path or "").strip()
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            return False
        image = _eai_platform_bootstrap.cv2.imread(path, _eai_platform_bootstrap.cv2.IMREAD_COLOR)
        if image is None:
            return False
        self._bagel_call_image_path = path
        h, w = image.shape[:2]
        if hasattr(self, "bagel_call_input_size_label"):
            self.bagel_call_input_size_label.setText(f"输入分辨率: {w}×{h}")
        pix = _eai_adapters_robot_geometry.cv2_to_qpixmap(image)
        if pix is not None and not pix.isNull():
            self.bagel_call_input_preview.setPixmap(
                pix.scaled(
                    self.bagel_call_input_preview.size(),
                    _eai_platform_bootstrap.Qt.KeepAspectRatio,
                    _eai_platform_bootstrap.Qt.SmoothTransformation,
                )
            )
            self.bagel_call_input_preview.setText("")
        tip = display_name or _eai_platform_bootstrap.os.path.basename(path)
        self.bagel_call_input_preview.setToolTip(f"{tip}\n{path}\n{w}×{h}")
        return True

    def _refresh_bagel_history_title_label(self) -> None:
        label = getattr(self, "bagel_history_title_label", None)
        if label is None:
            return
        if self._bagel_history_id and self._bagel_history_title:
            label.setText(f"当前: {self._bagel_history_title}")
        elif self._bagel_history_turns:
            label.setText("当前: 未命名会话（有未落盘轮次，点「保存」）")
        else:
            label.setText("当前: 新对话（未保存）")

    def _refresh_bagel_history_turns_list(self) -> None:
        lw = getattr(self, "bagel_history_turns_list", None)
        if lw is None:
            return
        lw.clear()
        for i, turn in enumerate(self._bagel_history_turns):
            task = str(turn.get("task") or "")
            task_lbl = _eai_infrastructure_ui_storage.BAGEL_TASK_LABELS.get(task, task or "?")
            prompt = " ".join(str(turn.get("prompt") or "").split())
            if len(prompt) > 28:
                prompt = prompt[:25] + "…"
            lat = turn.get("latency_s")
            try:
                lat_txt = f" {float(lat):.1f}s" if lat is not None else ""
            except (TypeError, ValueError):
                lat_txt = ""
            ok = bool(turn.get("ok", True))
            mark = "" if ok else "!"
            row = _eai_platform_bootstrap.QListWidgetItem(f"{i + 1}. {mark}{task_lbl}  {prompt}{lat_txt}")
            row.setData(_eai_platform_bootstrap.Qt.UserRole, i)
            tip_parts = [
                f"task={task}",
                f"prompt={turn.get('prompt') or ''}",
                f"in={turn.get('input_image') or ''}",
                f"out={turn.get('output_image') or ''}",
            ]
            row.setToolTip("\n".join(tip_parts))
            lw.addItem(row)

    def _load_bagel_web(self, url: str) -> None:
        view = self._ensure_bagel_web_view()
        if view is None:
            return
        try:
            page = view.page()
            try:
                page.setBackgroundColor(_eai_platform_bootstrap.QColor("#1a1a1a"))
            except Exception:
                pass
            view.setUrl(_eai_platform_bootstrap.QUrl(url))
            # 页面加载后再强制深色，覆盖 Gradio 未覆盖到的白底块
            _eai_platform_bootstrap.QTimer.singleShot(800, self._force_bagel_web_dark)
            _eai_platform_bootstrap.QTimer.singleShot(2500, self._force_bagel_web_dark)
            self._sync_bagel_gradio_embed()
        except Exception as exc:
            self._append_bagel_log(f"[warn] 嵌入页加载失败: {exc}")

    def _force_bagel_web_dark(self) -> None:
        if self.bagel_web_view is None:
            return
        js = (
            "(function(){"
            "try{"
            "document.documentElement.classList.add('dark');"
            "document.body.classList.add('dark');"
            "document.documentElement.style.background='#1a1a1a';"
            "document.body.style.background='#1a1a1a';"
            "var s=document.getElementById('eai-dark-override');"
            "if(!s){s=document.createElement('style');s.id='eai-dark-override';"
            "document.head.appendChild(s);}"
            "s.textContent='"
            "html,body,.gradio-container,.main,.app,.fillable,.contain,.wrap,"
            ".panel,.tabs,.tabitem,.form,.block,.image-container,.upload-container,"
            ".empty{background:#1a1a1a!important;background-color:#1a1a1a!important;"
            "color:#ececec!important;}"
            "textarea,input,select{background:#1e1e1e!important;color:#ececec!important;"
            "border-color:#555!important;}"
            ".tab-nav button{background:#252525!important;color:#ececec!important;}"
            "';"
            "}catch(e){}"
            "})();"
        )
        try:
            self.bagel_web_view.page().runJavaScript(js)
        except Exception:
            pass

