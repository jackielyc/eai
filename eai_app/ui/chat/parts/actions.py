"""ui.chat.parts.actions — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_chat_history_dialogs = module_ref('eai_app.ui.chat.history_dialogs')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

class ActionsMixin:
    def _blank_session_state(self) -> _eai_platform_bootstrap.Dict[str, object]:
        # 新会话继承当前模型/API，可在新 tab 里再改成别的
        state: _eai_platform_bootstrap.Dict[str, object] = {
            "messages": [],
            "history_id": "",
            "history_title": "",
            "lake_language_memory": _eai_configuration_chat.LAKE_DEFAULT_LANGUAGE_MEMORY,
            "traditional_chat_mode": bool(self._traditional_chat_mode),
            "history_html": "",
            "input_draft": "",
            "chat_attach_image_bgr": None,
            "chat_attach_image_path": "",
        }
        state.update(self._snapshot_llm_settings())
        return state

    def _snapshot_llm_settings(self) -> _eai_platform_bootstrap.Dict[str, object]:
        return {
            "provider": self.provider_combo.currentText(),
            "model": self.model_edit.text(),
            "api_base": self.api_base_edit.text(),
            "api_key": self.api_key_edit.text(),
            "enable_thinking": bool(self.thinking_check.isChecked()),
            "thinking_enabled": bool(self.thinking_check.isEnabled()),
            "attach_camera": bool(self.attach_camera_check.isChecked()),
            "system_prompt": self.system_prompt_edit.toPlainText(),
        }

    def _current_running_service(self) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, str]]:
        data = self.provider_combo.currentData()
        if isinstance(data, dict) and data.get("api_base") and data.get("model"):
            return {
                "label": str(data.get("label") or self.provider_combo.currentText()),
                "family": str(data.get("family") or ""),
                "api_base": str(data.get("api_base") or "").rstrip("/"),
                "model": str(data.get("model") or "").strip(),
                "api_key": str(data.get("api_key") or "EMPTY"),
            }
        return None

    def _apply_running_service(
        self, svc: _eai_platform_bootstrap.Dict[str, str], *, silent: bool = False
    ) -> None:
        api_base = str(svc.get("api_base") or "").rstrip("/")
        model = str(svc.get("model") or "").strip()
        api_key = str(svc.get("api_key") or "EMPTY")
        family = str(svc.get("family") or "")
        label = str(svc.get("label") or f"{family} · {model}")
        if api_base:
            self.api_base_edit.setText(api_base)
        if model:
            self.model_edit.setText(model)
        self.api_key_edit.setText(api_key or "EMPTY")

        vision = family in (
            "本地 Qwen",
            "远程 Qwen",
            "Ollama",
            "Hy-Embodied-VLM",
            "Hy-Embodied-RxBrain",
            "Bagel",
        ) or "vl" in model.lower()
        thinking = family == "Hy-Embodied-VLM"
        self.attach_camera_check.setEnabled(True)
        if vision:
            self.attach_camera_check.setChecked(True)
        self.thinking_check.setEnabled(thinking)
        if not thinking:
            self.thinking_check.setChecked(False)
        self._sync_config_from_ui()
        if not silent:
            self._append_system_line(
                f"已选择运行中服务: {label}  |  API: {api_base}"
            )

    def refresh_running_providers(
        self,
        *,
        prefer_api: _eai_platform_bootstrap.Optional[str] = None,
        prefer_model: _eai_platform_bootstrap.Optional[str] = None,
        notify: bool = False,
        silent_apply: bool = True,
    ) -> int:
        """刷新下拉框：只保留探测到的运行中推理服务。返回可用数量。"""
        prefer_api = (prefer_api or self.api_base_edit.text() or "").strip().rstrip("/")
        prefer_model = (prefer_model or self.model_edit.text() or "").strip()
        try:
            services = _eai_adapters_models_chat_client.discover_running_inference_services(
                timeout_s=1.0,
                extra_candidates=getattr(self, "_extra_inference_candidates", None),
            )
        except Exception as exc:
            services = []
            if notify:
                self._append_error_line(f"刷新运行中服务失败: {exc}")

        self._running_services = services
        self._provider_switch_guard = True
        try:
            self.provider_combo.clear()
            if not services:
                self.provider_combo.addItem("（无运行中的推理服务）", None)
                self.model_edit.setText("")
                self.send_btn.setEnabled(False)
                if notify:
                    self._append_system_line(
                        "未发现运行中的推理服务。请先在「测试」页启动本地/远程 Qwen，"
                        "或启动 Ollama / Hy-Embodied。"
                    )
                    self.status_message.emit("无运行中的推理服务")
            else:
                select_idx = 0
                for i, svc in enumerate(services):
                    self.provider_combo.addItem(svc["label"], svc)
                    if prefer_api and svc["api_base"].rstrip("/") == prefer_api.rstrip("/"):
                        if not prefer_model or svc["model"] == prefer_model:
                            select_idx = i
                        elif prefer_model.lower() in svc["model"].lower():
                            select_idx = i
                self.provider_combo.setCurrentIndex(select_idx)
                data = self.provider_combo.itemData(select_idx)
                if isinstance(data, dict):
                    self._apply_running_service(data, silent=silent_apply)
                if not self._busy:
                    self.send_btn.setEnabled(True)
                if notify:
                    shown = "、".join(s["label"] for s in services[:8])
                    suffix = "…" if len(services) > 8 else ""
                    self._append_system_line(
                        f"运行中推理服务 ({len(services)}): {shown}{suffix}"
                    )
                    self.status_message.emit(f"可用推理服务 {len(services)} 个")
        finally:
            self._provider_switch_guard = False
        self._sync_session_tab_title()
        return len(services)

    def _on_running_service_changed(self, _index: int) -> None:
        if getattr(self, "_provider_switch_guard", False):
            return
        svc = self._current_running_service()
        if svc is None:
            self.send_btn.setEnabled(False)
            return
        self._apply_running_service(svc, silent=False)
        self._sync_session_tab_title()

    def _restore_llm_settings(self, state: _eai_platform_bootstrap.Dict[str, object]) -> None:
        prefer_api = str(state.get("api_base") or "").strip()
        prefer_model = str(state.get("model") or "").strip()
        self.refresh_running_providers(
            prefer_api=prefer_api or None,
            prefer_model=prefer_model or None,
            notify=False,
            silent_apply=True,
        )
        # 若会话记录的服务仍在列表中，refresh 已选中；否则沿用当前可用服务
        if "attach_camera" in state:
            self.attach_camera_check.setChecked(bool(state.get("attach_camera")))

        provider = str(state.get("provider") or "").strip()
        thinking_enabled = state.get("thinking_enabled")
        if thinking_enabled is None:
            thinking_enabled = "Hy-Embodied-VLM" in provider
        family = ""
        svc = self._current_running_service()
        if svc:
            family = svc.get("family") or ""
        if family == "Hy-Embodied-VLM":
            thinking_enabled = True
        else:
            thinking_enabled = False
        self.thinking_check.setEnabled(bool(thinking_enabled))
        self.thinking_check.setChecked(
            bool(state.get("enable_thinking")) and bool(thinking_enabled)
        )

        sys_text = state.get("system_prompt")
        if isinstance(sys_text, str):
            self.system_prompt_edit.setPlainText(sys_text)

        # 保留会话里保存的 key（若有）
        saved_key = state.get("api_key")
        if isinstance(saved_key, str) and saved_key.strip():
            self.api_key_edit.setText(saved_key)

        self._sync_config_from_ui()

    def _snapshot_current_session(self) -> _eai_platform_bootstrap.Dict[str, object]:
        state: _eai_platform_bootstrap.Dict[str, object] = {
            "messages": list(self._messages),
            "history_id": self._history_id,
            "history_title": self._history_title,
            "lake_language_memory": self._lake_language_memory,
            "traditional_chat_mode": bool(self._traditional_chat_mode),
            "history_html": self.history_view.toHtml(),
            "input_draft": self.input_edit.toPlainText(),
            "chat_attach_image_bgr": self._chat_attach_image_bgr,
            "chat_attach_image_path": self._chat_attach_image_path,
        }
        state.update(self._snapshot_llm_settings())
        return state

    def _restore_attached_image_ui(
        self, image_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray], path: str
    ) -> None:
        self._chat_attach_image_bgr = image_bgr
        self._chat_attach_image_path = path if image_bgr is not None else ""
        if image_bgr is None:
            self.clear_image_btn.setEnabled(False)
            self.chat_image_label.setText(
                "未选场景图（测试页点选，或勾选附带相机图）"
            )
            self.chat_image_label.setToolTip("")
            self.chat_image_preview.clear()
            self.chat_image_preview.setText("预览")
            self.attached_image_changed.emit("")
            return
        self.clear_image_btn.setEnabled(not self._busy)
        name = _eai_platform_bootstrap.os.path.basename(path) if path else "场景图"
        h, w = image_bgr.shape[:2]
        self.chat_image_label.setText(f"已选场景: {name}  ({w}×{h})")
        self.chat_image_label.setToolTip(path)
        pix = _eai_adapters_robot_geometry.cv2_to_qpixmap(image_bgr)
        if pix is not None and not pix.isNull():
            scaled = pix.scaled(
                self.chat_image_preview.size(),
                _eai_platform_bootstrap.Qt.KeepAspectRatio,
                _eai_platform_bootstrap.Qt.SmoothTransformation,
            )
            self.chat_image_preview.setPixmap(scaled)
            self.chat_image_preview.setText("")
        self.attached_image_changed.emit(path)

    def _restore_session(self, state: _eai_platform_bootstrap.Dict[str, object]) -> None:
        raw_msgs = state.get("messages") or []
        self._messages = list(raw_msgs) if isinstance(raw_msgs, list) else []
        self._history_id = str(state.get("history_id") or "")
        self._history_title = str(state.get("history_title") or "")
        self._lake_language_memory = str(
            state.get("lake_language_memory") or _eai_configuration_chat.LAKE_DEFAULT_LANGUAGE_MEMORY
        )
        # 先恢复该会话的模型/API，再套传统对话模式（避免预设信号冲掉模型名）
        self._restore_llm_settings(state)
        trad = bool(state.get("traditional_chat_mode"))
        self._apply_traditional_chat_mode(trad, notify=False)
        if not trad:
            sys_text = state.get("system_prompt")
            if isinstance(sys_text, str):
                self.system_prompt_edit.setPlainText(sys_text)
                self._config.system_prompt = sys_text
                self._client = _eai_adapters_models_chat_client.LlmChatClient(self._config)
        html = str(state.get("history_html") or "")
        if html.strip():
            self.history_view.setHtml(html)
        else:
            self.history_view.clear()
            self._render_messages_to_view()
        self.input_edit.setPlainText(str(state.get("input_draft") or ""))
        img = state.get("chat_attach_image_bgr")
        path = str(state.get("chat_attach_image_path") or "")
        image_bgr = img if isinstance(img, _eai_platform_bootstrap.np.ndarray) else None
        self._restore_attached_image_ui(image_bgr, path)
        self._refresh_history_title_label()

    def _current_session_display_title(self) -> str:
        title = (self._history_title or "").strip()
        if title:
            return title
        if self._conversation_message_count() == 0:
            return "新对话"
        return _eai_infrastructure_ui_storage.default_chat_history_title(self._messages)

    def _sync_session_tab_title(self) -> None:
        idx = self.session_tabs.currentIndex()
        if idx < 0:
            return
        full = self._current_session_display_title()
        model = (self.model_edit.text().strip() or self._config.model or "").strip()
        short = full if len(full) <= 14 else full[:13] + "…"
        self.session_tabs.setTabText(idx, short)
        tip = f"{full}\n模型: {model}" if model else full
        self.session_tabs.setTabToolTip(idx, tip)

    def _on_new_session_clicked(self) -> None:
        if self._busy:
            self.status_message.emit("请等待当前回复完成后再新开对话")
            return
        cur = self.session_tabs.currentIndex()
        if 0 <= cur < len(self._sessions):
            self._sessions[cur] = self._snapshot_current_session()
        blank = self._blank_session_state()
        self._sessions.append(blank)
        self._session_switch_guard = True
        try:
            new_idx = self.session_tabs.addTab("新对话")
            self.session_tabs.setCurrentIndex(new_idx)
            self._session_index = new_idx
        finally:
            self._session_switch_guard = False
        self._restore_session(blank)
        n = self.refresh_running_providers(notify=False, silent_apply=True)
        model = (self.model_edit.text().strip() or self._config.model or "").strip()
        api = (self.api_base_edit.text().strip() or self._config.api_base or "").strip()
        if n <= 0:
            self._append_system_line(
                "已新开对话，但当前没有运行中的推理服务；请先部署并启动后再选择。"
            )
        else:
            self._append_system_line(
                f"已新开对话 | 可选运行中服务 {n} 个 | 当前: {model}  |  API: {api}"
            )
        self._sessions[new_idx] = self._snapshot_current_session()
        self._sync_session_tab_title()
        self._focus_chat_input()
        self.status_message.emit(
            "已新开对话（仅可选择运行中的推理服务）"
            if n > 0
            else "已新开对话：无运行中推理服务"
        )

    def _on_session_tab_changed(self, index: int) -> None:
        if self._session_switch_guard:
            return
        if index < 0:
            return
        if self._busy:
            self._session_switch_guard = True
            try:
                self.session_tabs.setCurrentIndex(self._session_index)
            finally:
                self._session_switch_guard = False
            self.status_message.emit("请等待当前回复完成后再切换对话")
            return
        prev = self._session_index
        if 0 <= prev < len(self._sessions) and prev != index:
            self._sessions[prev] = self._snapshot_current_session()
        if 0 <= index < len(self._sessions):
            self._restore_session(self._sessions[index])
            self._session_index = index
            self._focus_chat_input()

    def _on_session_tab_close(self, index: int) -> None:
        if self._busy:
            self.status_message.emit("请等待当前回复完成后再关闭对话")
            return
        if index < 0 or index >= self.session_tabs.count():
            return
        if self.session_tabs.count() <= 1:
            self._clear_chat()
            if self._sessions:
                self._sessions[0] = self._snapshot_current_session()
            self._sync_session_tab_title()
            return
        self._session_switch_guard = True
        try:
            cur = self.session_tabs.currentIndex()
            if 0 <= cur < len(self._sessions):
                self._sessions[cur] = self._snapshot_current_session()
            if 0 <= index < len(self._sessions):
                del self._sessions[index]
            self.session_tabs.removeTab(index)
            new_cur = self.session_tabs.currentIndex()
            if new_cur < 0 and self.session_tabs.count() > 0:
                new_cur = 0
                self.session_tabs.setCurrentIndex(0)
            self._session_index = new_cur
            if 0 <= new_cur < len(self._sessions):
                self._restore_session(self._sessions[new_cur])
        finally:
            self._session_switch_guard = False
        self._focus_chat_input()

    def _on_session_tab_moved(self, from_index: int, to_index: int) -> None:
        if from_index == to_index:
            return
        if not (0 <= from_index < len(self._sessions) and 0 <= to_index < len(self._sessions)):
            self._session_index = self.session_tabs.currentIndex()
            return
        item = self._sessions.pop(from_index)
        self._sessions.insert(to_index, item)
        self._session_index = self.session_tabs.currentIndex()

    def set_camera_frame_provider(
        self,
        provider: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray]]]],
    ) -> None:
        self._camera_frame_provider = provider

    def _apply_traditional_chat_mode(self, enabled: bool, *, notify: bool = True) -> None:
        self._traditional_chat_mode = bool(enabled)
        if self._traditional_chat_mode:
            self._config.system_prompt = ""
            self.system_prompt_edit.setPlainText("")
            self.input_edit.setPlaceholderText(_eai_configuration_chat.TRADITIONAL_CHAT_INPUT_PLACEHOLDER)
            self.traditional_chat_btn.setStyleSheet(
                f"QPushButton {{ background-color: #2a4a6a; color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; }}"
            )
            if notify:
                self._append_system_line(
                    "已切换为传统对话：System prompt 已清空，后续不再注入 Lake 编排提示词"
                )
                self.status_message.emit("传统对话模式")
        else:
            self.traditional_chat_btn.setStyleSheet("")
            if not self.system_prompt_edit.toPlainText().strip():
                self.system_prompt_edit.setPlainText(_eai_configuration_chat.LAKE_ORCHESTRATOR_SYSTEM_PROMPT)
                self._config.system_prompt = _eai_configuration_chat.LAKE_ORCHESTRATOR_SYSTEM_PROMPT
            self._refresh_system_lang_btn()

    def _on_traditional_chat_clicked(self) -> None:
        self._apply_traditional_chat_mode(True)
        try:
            _eai_infrastructure_ui_storage.save_chat_user_settings({"system_prompt": ""})
        except Exception as exc:
            self._append_system_line(f"保存空 System prompt 失败: {exc}")

    def _on_settings_toggled(self, checked: bool) -> None:
        self.settings_panel.setVisible(bool(checked))
        self.settings_toggle_btn.setText("收起" if checked else "设置")
        if checked:
            self.system_prompt_edit.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)
            _eai_ui_widgets_inputs._schedule_fcitx_restore(self.system_prompt_edit)
        else:
            _eai_platform_bootstrap.QTimer.singleShot(0, self._focus_chat_input)
            _eai_ui_widgets_inputs._schedule_fcitx_restore(self.input_edit)

    def _on_probe_clicked(self) -> None:
        if self._busy:
            return
        self._append_system_line("正在刷新运行中的推理服务列表…")
        n = self.refresh_running_providers(notify=True, silent_apply=True)
        if n <= 0:
            return
        self._sync_config_from_ui()
        self._set_busy(True)
        self.probe_btn.setText("…")
        self._append_system_line(
            f"正在探测当前服务: {self._config.api_base}  model={self._config.model}"
        )
        self.status_message.emit("正在探测连接的模型…")

        def _work() -> None:
            try:
                result = self._client.probe(timeout_s=6.0)
            except Exception as exc:
                result = {
                    "ok": False,
                    "kind": "探测失败",
                    "api_base": self._config.api_base,
                    "configured_model": self._config.model,
                    "models": [],
                    "details": [],
                    "error": str(exc),
                }
            self._probe_bridge.finished.emit(result)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_probe_finished(self, result: object) -> None:
        self._set_busy(False)
        self.probe_btn.setText("探测")
        data = result if isinstance(result, dict) else {}
        ok = bool(data.get("ok"))
        kind = str(data.get("kind") or "未知")
        models = [str(m) for m in (data.get("models") or []) if str(m).strip()]
        configured = str(data.get("configured_model") or "")
        matched = bool(data.get("configured_model_matched"))
        details = [str(x) for x in (data.get("details") or [])]
        err = str(data.get("error") or "")

        lines = [
            f"服务类型: {kind}",
            f"API: {data.get('api_base') or self._config.api_base}",
        ]
        if configured:
            mark = "✓ 匹配" if matched else ("未在列表中确认" if models else "待确认")
            lines.append(f"当前配置模型: {configured}（{mark}）")
        if models:
            show = ", ".join(models[:8])
            if len(models) > 8:
                show += f" …共{len(models)}个"
            lines.append(f"可用模型: {show}")
        else:
            lines.append("可用模型: （未列出）")
        for d in details[:6]:
            lines.append(f"- {d}")
        if err:
            lines.append(f"错误: {err}")

        summary = "\n".join(lines)
        if ok:
            self._append_system_line("模型探测完成:\n" + summary)
            self.status_message.emit(f"探测完成: {kind}")
            # 若只发现一个模型且与配置不同，提示可切换
            if len(models) == 1 and configured and models[0] != configured:
                reply = _eai_platform_bootstrap.QMessageBox.question(
                    self,
                    "探测模型",
                    f"检测到服务类型: {kind}\n"
                    f"可用模型: {models[0]}\n"
                    f"当前配置: {configured}\n\n"
                    "是否切换到检测到的模型？",
                    _eai_platform_bootstrap.QMessageBox.Yes | _eai_platform_bootstrap.QMessageBox.No,
                    _eai_platform_bootstrap.QMessageBox.Yes,
                )
                if reply == _eai_platform_bootstrap.QMessageBox.Yes:
                    self.refresh_running_providers(
                        prefer_api=str(data.get("api_base") or self._config.api_base),
                        prefer_model=models[0],
                        notify=False,
                        silent_apply=True,
                    )
                    self._append_system_line(f"已切换到运行中模型: {models[0]}")
            else:
                _eai_platform_bootstrap.QMessageBox.information(self, "探测模型", summary)
        else:
            self._append_error_line("模型探测失败:\n" + summary)
            self.status_message.emit(f"探测失败: {err or kind}")
            _eai_platform_bootstrap.QMessageBox.warning(self, "探测模型", summary)

    def apply_local_ollama_preset(self, silent: bool = False) -> None:
        """切换到本地 Ollama（若正在运行则选入下拉框）。"""
        n = self.refresh_running_providers(
            prefer_api=_eai_configuration_perception.OLLAMA_API_BASE_DEFAULT,
            prefer_model="qwen3.5:4b",
            notify=not silent,
            silent_apply=silent,
        )
        if n <= 0 and not silent:
            self._append_system_line(
                f"Ollama 未在运行（期望 {_eai_configuration_perception.OLLAMA_API_BASE_DEFAULT}）"
            )

    def apply_local_qwen_service_preset(
        self,
        api_base: _eai_platform_bootstrap.Optional[str] = None,
        model_id: _eai_platform_bootstrap.Optional[str] = None,
        silent: bool = False,
    ) -> None:
        """切换到本地 Qwen 推理服务（须已在运行）。"""
        base = (api_base or _eai_adapters_models_qwen_local.resolve_local_qwen_viewer_api_base()).rstrip("/")
        mid = (model_id or "").strip()
        if not mid:
            info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(base) or {}
            mid = str(info.get("model") or "").strip() or _eai_configuration_policies.LOCAL_QWEN_MODEL_ID
        n = self.refresh_running_providers(
            prefer_api=base,
            prefer_model=mid,
            notify=False,
            silent_apply=True,
        )
        if n <= 0:
            if not silent:
                self._append_system_line(
                    f"本地 Qwen 未在运行，无法切换（{base}）"
                )
            return
        if not silent:
            self._append_system_line(
                f"已切换对话 API 为本地 Qwen 服务 {base}  model={mid}"
            )

    def apply_remote_qwen_service_preset(
        self,
        api_base: _eai_platform_bootstrap.Optional[str] = None,
        model_id: _eai_platform_bootstrap.Optional[str] = None,
        silent: bool = False,
    ) -> None:
        """切换到远程 Qwen（SSH 隧道）推理服务（须已在运行）。"""
        base = (api_base or _eai_configuration_policies.REMOTE_QWEN_API_BASE_DEFAULT).rstrip("/")
        mid = (model_id or "").strip()
        if not mid:
            info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(base) or {}
            mid = str(info.get("model") or "").strip() or "qwen3.5-35b-a3b"
        n = self.refresh_running_providers(
            prefer_api=base,
            prefer_model=mid,
            notify=False,
            silent_apply=True,
        )
        if n <= 0:
            if not silent:
                self._append_system_line(
                    f"远程 Qwen 未在运行，无法切换（{base}）"
                )
            return
        if not silent:
            self._append_system_line(
                f"已切换对话 API 为远程 Qwen 服务 {base}  model={mid}"
            )

    def _apply_provider_preset(self, name: str, silent: bool = False) -> None:
        preset = _eai_configuration_chat.LLM_PROVIDER_PRESETS.get(name)
        if preset is None or name == "自定义":
            return
        api_base, model, default_key = preset
        if api_base:
            self.api_base_edit.setText(api_base)
        if name == _eai_configuration_policies.LOCAL_QWEN_CHAT_PRESET_NAME:
            info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api_base) or {}
            live_model = str(info.get("model") or "").strip()
            self.model_edit.setText(live_model or model)
        elif name == _eai_configuration_policies.REMOTE_QWEN_CHAT_PRESET_NAME:
            info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api_base) or {}
            live_model = str(info.get("model") or "").strip()
            self.model_edit.setText(live_model or model)
        elif model:
            self.model_edit.setText(model)
        if default_key and not self.api_key_edit.text().strip():
            self.api_key_edit.setText(default_key)
        vision = name in _eai_configuration_chat.LLM_VISION_PROVIDER_NAMES
        thinking = name in _eai_configuration_chat.LLM_THINKING_PROVIDER_NAMES
        self.attach_camera_check.setEnabled(True)
        if vision:
            self.attach_camera_check.setChecked(True)
        self.thinking_check.setEnabled(thinking)
        if not thinking:
            self.thinking_check.setChecked(False)
        if not silent:
            hint = ""
            if name == "Hy-Embodied-VLM-1.0":
                hint = "（需: bash run_hy_embodied_vlm.sh，约 4×80GB GPU）"
            elif name == "Hy-Embodied-RxBrain-1.0":
                hint = "（需: bash run_hy_rxbrain.sh，约 1×GPU + 权重）"
            elif name == _eai_configuration_policies.LOCAL_QWEN_CHAT_PRESET_NAME:
                hint = "（需: 测试 Tab 选择模型并「启动推理服务」）"
            elif name == _eai_configuration_policies.REMOTE_QWEN_CHAT_PRESET_NAME:
                hosts = " / ".join(h for h, _ in _eai_configuration_policies.REMOTE_QWEN_HOSTS)
                hint = f"（需: 测试 Tab 部署位置选远程，如 {hosts}）"
            self._append_system_line(f"已切换预设: {name}{hint}")

    def _on_provider_preset_changed(self, name: str) -> None:
        # 兼容旧调用：改为按运行中服务刷新并匹配名称
        self.refresh_running_providers(notify=False, silent_apply=True)
        idx = self.provider_combo.findText(name)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self._sync_session_tab_title()

    def set_attached_image_from_path(
        self, path: str, display_name: str = ""
    ) -> bool:
        """用测试页场景图路径设置为对话附图；成功返回 True。"""
        image = _eai_platform_bootstrap.cv2.imread(path, _eai_platform_bootstrap.cv2.IMREAD_COLOR)
        if image is None or _eai_platform_bootstrap.np.asarray(image).size == 0:
            self._append_error_line(f"无法读取场景图: {path}")
            return False
        self._chat_attach_image_bgr = image
        self._chat_attach_image_path = path
        self.clear_image_btn.setEnabled(not self._busy)
        name = display_name or _eai_platform_bootstrap.os.path.basename(path)
        h, w = image.shape[:2]
        self.chat_image_label.setText(f"已选场景: {name}  ({w}×{h})")
        self.chat_image_label.setToolTip(path)
        pix = _eai_adapters_robot_geometry.cv2_to_qpixmap(image)
        if pix is not None and not pix.isNull():
            scaled = pix.scaled(
                self.chat_image_preview.size(),
                _eai_platform_bootstrap.Qt.KeepAspectRatio,
                _eai_platform_bootstrap.Qt.SmoothTransformation,
            )
            self.chat_image_preview.setPixmap(scaled)
            self.chat_image_preview.setText("")
        self._append_system_line(f"已点选场景图: {name}")
        self.attached_image_changed.emit(path)
        return True

    def get_attached_image_path(self) -> str:
        return str(self._chat_attach_image_path or "").strip()

    def get_last_role_text(self, role: str = "assistant") -> str:
        """取当前会话中最近一条指定角色的纯文本（跳过 BAGEL_IMAGE 元数据行）。"""
        want = str(role or "").strip()
        for msg in reversed(self._messages):
            if str(msg.get("role") or "") != want:
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            text = content.strip()
            if text.startswith("BAGEL_IMAGE::"):
                _first, _, rest = text.partition("\n")
                text = rest.strip()
            if text:
                return text
        return ""

    def _on_clear_chat_image_clicked(self, *, notify: bool = True) -> None:
        had_image = self._chat_attach_image_bgr is not None
        self._chat_attach_image_bgr = None
        self._chat_attach_image_path = ""
        self.clear_image_btn.setEnabled(False)
        self.chat_image_label.setText(
            "未选场景图（测试页点选，或勾选附带相机图）"
        )
        self.chat_image_label.setToolTip("")
        self.chat_image_preview.clear()
        self.chat_image_preview.setText("预览")
        if had_image:
            self.attached_image_changed.emit("")
            if notify:
                self._append_system_line("已清除所选场景图")

    def _rebuild_chat_input(self, text: _eai_platform_bootstrap.Optional[str] = None) -> "ChatInputEdit":
        """替换输入框，强制 fcitx 重新挂 IC。"""
        pass
        if getattr(self, "_ime_rebuilding", False):
            return self.input_edit
        self._ime_rebuilding = True
        try:
            old = self.input_edit
            if text is None:
                text = old.toPlainText()
            placeholder = old.placeholderText()
            style = old.styleSheet()
            height = old.height()
            new = _eai_ui_chat_history_dialogs.ChatInputEdit(self)
            new._chat_panel = self
            new.setPlaceholderText(placeholder)
            new.setFixedHeight(height if height > 0 else 100)
            new.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Fixed)
            new.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
            new.setStyleSheet(style)
            try:
                new.document().setDocumentMargin(4)
            except Exception:
                pass
            new.setPlainText(text)
            cursor = new.textCursor()
            cursor.movePosition(cursor.End)
            new.setTextCursor(cursor)
            row = getattr(self, "_input_row", None)
            if row is not None:
                row.replaceWidget(old, new)
            old.deleteLater()
            self.input_edit = new
            bar = getattr(self, "input_resize_bar", None)
            if bar is not None:
                bar._log = new
            self._ime_dirty_from_history = False
            _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET = new
            new.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)
            return new
        finally:
            self._ime_rebuilding = False

    def _focus_chat_input(self) -> None:
        self.input_edit.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
        self.input_edit.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)
        _eai_ui_widgets_inputs._notify_ime_widget(self.input_edit)

    def _sync_config_from_ui(self) -> None:
        self._config.api_base = self.api_base_edit.text().strip() or _eai_configuration_chat.LLM_API_BASE_DEFAULT
        self._config.model = self.model_edit.text().strip() or _eai_configuration_chat.LLM_MODEL_DEFAULT
        key = self.api_key_edit.text().strip()
        if not key:
            key = _eai_platform_bootstrap.os.environ.get(_eai_configuration_chat.LLM_API_KEY_ENV, "").strip()
        self._config.api_key = key
        self._config.enable_thinking = bool(self.thinking_check.isChecked())
        system = self.system_prompt_edit.toPlainText().strip()
        if system:
            self._traditional_chat_mode = False
            self.traditional_chat_btn.setStyleSheet("")
            self.input_edit.setPlaceholderText(_eai_configuration_chat.LAKE_CHAT_INPUT_PLACEHOLDER)
            self._config.system_prompt = system
        elif self._traditional_chat_mode:
            self._config.system_prompt = ""
        else:
            self._config.system_prompt = _eai_configuration_chat.LAKE_ORCHESTRATOR_SYSTEM_PROMPT
        self._client = _eai_adapters_models_chat_client.LlmChatClient(self._config)

    def _on_save_system_prompt_clicked(self) -> None:
        self._sync_config_from_ui()
        if self._config.system_prompt.strip():
            self._apply_traditional_chat_mode(False, notify=False)
        else:
            self._apply_traditional_chat_mode(True, notify=False)
        lang = _eai_configuration_chat.lake_orchestrator_prompt_lang(self._config.system_prompt)
        try:
            path = _eai_infrastructure_ui_storage.save_chat_user_settings(
                {
                    "system_prompt": self._config.system_prompt,
                    "system_prompt_lang": lang,
                }
            )
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "保存 System", f"保存失败: {exc}")
            return
        preview = self._config.system_prompt.replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:57] + "…"
        aligned = _eai_configuration_chat.lake_user_output_instruction(self._config.system_prompt)
        self._append_system_line(f"已保存 System prompt: {preview}")
        if aligned:
            self._append_system_line(f"User 输出要求已对齐: {aligned}")
        else:
            self._append_system_line("User 不再附加「请输出全部子任务…」尾句")
        self._refresh_system_lang_btn()
        self.status_message.emit(f"System prompt 已保存: {path}")

    def _on_reset_system_prompt_clicked(self) -> None:
        self._apply_traditional_chat_mode(False, notify=False)
        self.system_prompt_edit.setPlainText(_eai_configuration_chat.LAKE_ORCHESTRATOR_SYSTEM_PROMPT_TRAINING)
        self._sync_config_from_ui()
        self._refresh_system_lang_btn()
        self._append_system_line(
            "已恢复训练默认 System（三行版）；user 将自动对齐为「技能/子任务/记忆」"
        )

    def _refresh_system_lang_btn(self) -> None:
        if not hasattr(self, "system_lang_btn"):
            return
        lang = _eai_configuration_chat.lake_orchestrator_prompt_lang(
            self.system_prompt_edit.toPlainText()
            if hasattr(self, "system_prompt_edit")
            else self._config.system_prompt
        )
        # 按钮文案 = 下一次将切换到的状态
        if lang == "zh":
            self.system_lang_btn.setText("English")
            self.system_lang_btn.setToolTip(
                "当前为中文 system prompt；点击切换为 English 完整编排\n"
                "（Skill / Current / Previous / Next）"
            )
            placeholder = _eai_configuration_chat.LAKE_CHAT_INPUT_PLACEHOLDER
        elif lang == "en":
            self.system_lang_btn.setText("EN-2L")
            self.system_lang_btn.setToolTip(
                "当前为 English 完整版；点击切换为两行简版\n"
                "（Given a scene image… All subtasks, then Current subtask）\n"
                "对齐 hermas_sys2_current_only_*-en"
            )
            placeholder = _eai_configuration_chat.LAKE_CHAT_INPUT_PLACEHOLDER_EN
        else:
            self.system_lang_btn.setText("中文")
            self.system_lang_btn.setToolTip(
                "当前为 EN 两行简版（All subtasks + Current subtask）；\n"
                "点击切换回中文编排提示词"
            )
            placeholder = _eai_configuration_chat.LAKE_CHAT_INPUT_PLACEHOLDER_EN
        if (
            not self._traditional_chat_mode
            and hasattr(self, "input_edit")
            and self.input_edit is not None
        ):
            self.input_edit.setPlaceholderText(placeholder)

    def _on_toggle_system_prompt_lang(self) -> None:
        self._apply_traditional_chat_mode(False, notify=False)
        cur = _eai_configuration_chat.lake_orchestrator_prompt_lang(self.system_prompt_edit.toPlainText())
        # 中文 → English 完整 → EN 两行简版 → 中文
        if cur == "zh":
            new_lang = "en"
        elif cur == "en":
            new_lang = "en2"
        else:
            new_lang = "zh"
        prompt = _eai_configuration_chat.lake_orchestrator_system_prompt_for_lang(new_lang)
        self.system_prompt_edit.setPlainText(prompt)
        self._config.system_prompt = prompt
        # 切换语言时重置「上一个子任务」为该语言默认首步
        self._lake_language_memory = _eai_configuration_chat.lake_default_previous_subtask(new_lang)
        self._sync_config_from_ui()
        self._refresh_system_lang_btn()
        label = {
            "en": "English 完整版",
            "en2": "EN 两行简版（All/Current）",
            "zh": "中文",
        }.get(new_lang, new_lang)
        self._append_system_line(
            f"已切换 System prompt 为 {label}；user 仅包装「任务 / Task」行"
        )
        try:
            _eai_infrastructure_ui_storage.save_chat_user_settings(
                {
                    "system_prompt": prompt,
                    "system_prompt_lang": new_lang,
                }
            )
        except Exception as exc:
            self._append_system_line(f"保存语言设置失败: {exc}")

