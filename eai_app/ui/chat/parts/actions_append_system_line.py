"""ui.chat.parts.actions append system line — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_chat_history_dialogs = module_ref('eai_app.ui.chat.history_dialogs')

class ActionsAppendSystemLineMixin:
    def _append_system_line(self, text: str) -> None:
        self.history_view.append(
            f'<p style="margin:4px 0 10px 0;">'
            f'<span style="color:{_eai_configuration_policies.UI_TEXT_MUTED};">[系统] {{text}}</span></p>'
        )
        self._scroll_history_to_bottom()

    def _scroll_history_to_bottom(self) -> None:
        """滚到底部并留出一点底边距，避免最后一行被裁切。"""
        view = self.history_view

        def _go() -> None:
            try:
                cursor = view.textCursor()
                cursor.movePosition(cursor.End)
                view.setTextCursor(cursor)
                view.ensureCursorVisible()
                bar = view.verticalScrollBar()
                bar.setValue(bar.maximum())
            except Exception:
                pass

        _eai_platform_bootstrap.QTimer.singleShot(0, _go)
        _eai_platform_bootstrap.QTimer.singleShot(30, _go)

    def _append_prompt_dump(self, messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]]) -> None:
        """在对话框中打印即将发送的全部提示词。"""
        has_image = False
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and str(part.get("type") or "") in (
                        "image_url",
                        "image",
                    ):
                        has_image = True
                        break
            if has_image:
                break
        timeout_s = (
            _eai_configuration_chat.LLM_CHAT_VISION_TIMEOUT_S if has_image else _eai_configuration_chat.LLM_CHAT_TIMEOUT_S
        )
        dump = _eai_adapters_models_chat_client.format_chat_prompt_dump(
            messages,
            api_base=self._config.api_base,
            model=self._config.model,
            temperature=0.2,
            max_tokens=int(self._config.max_tokens),
            enable_thinking=bool(self._config.enable_thinking),
            timeout_s=timeout_s,
        )
        safe = _eai_adapters_models_chat_client._html_escape(dump).replace("\n", "<br>")
        self.history_view.append(
            f'<pre style="margin:8px 0 14px 0; padding:8px; white-space:pre-wrap; '
            f'word-wrap:break-word; color:{_eai_configuration_policies.UI_TEXT_MUTED}; '
            f'background:rgba(127,127,127,0.12); border-radius:6px; '
            f'font-size:11px; font-family:monospace;">{safe}</pre>'
        )
        self._scroll_history_to_bottom()

    def _meta_span(self, ts: _eai_platform_bootstrap.Optional[str] = None, latency_s: _eai_platform_bootstrap.Optional[float] = None) -> str:
        parts = [_eai_infrastructure_ui_storage.format_chat_timestamp(ts)]
        lat = _eai_infrastructure_ui_storage.format_chat_latency(latency_s)
        if lat:
            parts.append(f"耗时 {lat}")
        meta = " · ".join(parts)
        return (
            f'<span style="color:{_eai_configuration_policies.UI_TEXT_MUTED}; font-size:11px; margin-left:8px;">'
            f"{_eai_adapters_models_chat_client._html_escape(meta)}</span>"
        )

    def _append_user_line(self, text: str, ts: _eai_platform_bootstrap.Optional[str] = None) -> None:
        safe = _eai_adapters_models_chat_client._html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0 12px 0;">'
            f'<b style="color:#7ec8ff;">你:</b>{self._meta_span(ts)}'
            f"<br>{safe}</p>"
        )
        self._scroll_history_to_bottom()

    def _append_assistant_line(
        self,
        text: str,
        ts: _eai_platform_bootstrap.Optional[str] = None,
        latency_s: _eai_platform_bootstrap.Optional[float] = None,
    ) -> None:
        image_html = ""
        body = text
        if text.startswith("BAGEL_IMAGE::"):
            first, _, rest = text.partition("\n")
            path = first.split("::", 1)[-1].strip()
            body = rest
            if path and _eai_platform_bootstrap.os.path.isfile(path):
                url = _eai_platform_bootstrap.QUrl.fromLocalFile(_eai_platform_bootstrap.os.path.abspath(path)).toString()
                image_html = f'<br><img src="{url}" width="380" />'
        safe = _eai_adapters_models_chat_client._html_escape(body).replace("\n", "<br>") if body else ""
        self.history_view.append(
            f'<p style="margin:6px 0 12px 0;">'
            f'<b style="color:#50fa7b;">AI:</b>{self._meta_span(ts, latency_s)}'
            f"<br>{safe}{image_html}</p>"
        )
        self._scroll_history_to_bottom()

    def _append_error_line(
        self,
        text: str,
        ts: _eai_platform_bootstrap.Optional[str] = None,
        latency_s: _eai_platform_bootstrap.Optional[float] = None,
    ) -> None:
        safe = _eai_adapters_models_chat_client._html_escape(text).replace("\n", "<br>")
        self.history_view.append(
            f'<p style="margin:6px 0 12px 0;">'
            f'<b style="color:{_eai_configuration_policies.UI_ACCENT_RED};">错误:</b>{{self._meta_span(ts, latency_s)}}'
            f"<br>{safe}</p>"
        )
        self._scroll_history_to_bottom()

    def _trim_history(self) -> None:
        # 只保留最近的 user/assistant 轮次
        keep = [
            m
            for m in self._messages
            if str(m.get("role") or "") in ("user", "assistant")
        ]
        if len(keep) > _eai_configuration_chat.LLM_CHAT_MAX_HISTORY:
            keep = keep[-_eai_configuration_chat.LLM_CHAT_MAX_HISTORY:]
        self._messages = keep

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        has_svc = self._current_running_service() is not None
        self.send_btn.setEnabled((not busy) and has_svc)
        self.input_edit.setReadOnly(busy)
        self.save_btn.setEnabled(not busy)
        self.history_btn.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.probe_btn.setEnabled(not busy)
        self.new_session_btn.setEnabled(not busy)
        self.session_tabs.setEnabled(not busy)
        self.provider_combo.setEnabled(not busy)
        self.clear_image_btn.setEnabled(
            (not busy) and self._chat_attach_image_bgr is not None
        )
        self.send_btn.setText("思考中…" if busy else "发送")
        if not busy:
            _eai_platform_bootstrap.QTimer.singleShot(0, self._focus_chat_input)

    def _refresh_history_title_label(self) -> None:
        if self._history_id and self._history_title:
            self.history_title_label.setText(
                f"当前: {self._history_title}  (已保存)"
            )
        elif self._history_title:
            self.history_title_label.setText(f"当前: {self._history_title}")
        else:
            self.history_title_label.setText("当前: 新对话（未保存）")
        self._sync_session_tab_title()

    def _conversation_message_count(self) -> int:
        return sum(
            1
            for m in self._messages
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and str(m.get("content") or "").strip()
        )

    def _render_messages_to_view(self) -> None:
        self.history_view.clear()
        for msg in self._messages:
            role = str(msg.get("role") or "")
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "system":
                continue
            ts = str(msg.get("ts") or "") or None
            latency_raw = msg.get("latency_s")
            latency_s: _eai_platform_bootstrap.Optional[float] = None
            if isinstance(latency_raw, (int, float)):
                latency_s = float(latency_raw)
            if role == "user":
                self._append_user_line(content, ts=ts)
            elif role == "assistant":
                self._append_assistant_line(content, ts=ts, latency_s=latency_s)

    def _on_save_chat_clicked(self) -> None:
        if self._conversation_message_count() == 0:
            _eai_platform_bootstrap.QMessageBox.information(self, "保存对话", "当前没有可保存的对话内容")
            return
        self._sync_config_from_ui()
        title = (self._history_title or _eai_infrastructure_ui_storage.default_chat_history_title(self._messages)).strip()
        if not title:
            title = _eai_platform_bootstrap.time.strftime("对话 %Y-%m-%d %H:%M")
        record: _eai_platform_bootstrap.Dict[str, object] = {
            "id": self._history_id or _eai_platform_bootstrap.uuid.uuid4().hex[:12],
            "title": title,
            "created_at": "",
            "model": self._config.model,
            "api_base": self._config.api_base,
            "system_prompt": self._config.system_prompt,
            "messages": _eai_infrastructure_ui_storage.serialize_chat_messages(self._messages),
        }
        if self._history_id:
            try:
                old = _eai_infrastructure_ui_storage.load_chat_history(self._history_id)
                record["created_at"] = str(old.get("created_at") or "")
            except Exception:
                pass
        try:
            path = _eai_infrastructure_ui_storage.save_chat_history_record(record)
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "保存对话", f"保存失败: {exc}")
            return
        self._history_id = str(record["id"])
        self._history_title = title
        self._refresh_history_title_label()
        self._append_system_line(f"已保存对话: {title}")
        self.status_message.emit(f"对话已保存: {path}")

    def _on_history_clicked(self) -> None:
        existing = getattr(self, "_history_dialog", None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing.title_edit.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)
                    return
                existing.deleteLater()
            except Exception:
                pass
        dlg = _eai_ui_chat_history_dialogs.ChatHistoryDialog(self)
        self._history_dialog = dlg

        def _on_finished(result: int) -> None:
            self._history_dialog = None
            accepted = result == int(_eai_platform_bootstrap.QDialog.Accepted)
            hid = dlg.selected_id if accepted else ""
            dlg.deleteLater()
            if accepted and hid:
                self._load_history_by_id(hid)
                return
            if not accepted and self._history_id:
                try:
                    _eai_infrastructure_ui_storage.load_chat_history(self._history_id)
                except Exception:
                    self._history_id = ""
                    self._history_title = ""
                    self._refresh_history_title_label()
            self._focus_chat_input()

        dlg.finished.connect(_on_finished)
        dlg.show()
        dlg.raise_()
        _eai_platform_bootstrap.QTimer.singleShot(0, lambda: dlg.title_edit.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason))

    def _load_history_by_id(self, hid: str) -> None:
        hid = str(hid or "").strip()
        if not hid:
            return
        try:
            data = _eai_infrastructure_ui_storage.load_chat_history(hid)
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "历史对话", f"加载失败: {exc}")
            self._focus_chat_input()
            return
        raw_msgs = data.get("messages") or []
        messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
        if isinstance(raw_msgs, list):
            for msg in raw_msgs:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role") or "")
                content = msg.get("content")
                if role not in ("user", "assistant") or not isinstance(content, str):
                    continue
                item: _eai_platform_bootstrap.Dict[str, object] = {"role": role, "content": content}
                ts = msg.get("ts")
                if isinstance(ts, str) and ts.strip():
                    item["ts"] = ts.strip()
                latency = msg.get("latency_s")
                if isinstance(latency, (int, float)):
                    item["latency_s"] = float(latency)
                messages.append(item)
        draft = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "user":
                continue
            content = messages[i].get("content")
            if isinstance(content, str) and content.strip():
                draft = content
                messages = messages[:i]
            break
        self._messages = messages
        self._trim_history()
        self._lake_language_memory = _eai_configuration_chat.LAKE_DEFAULT_LANGUAGE_MEMORY
        for m in reversed(messages):
            if m.get("role") != "assistant":
                continue
            mem = _eai_configuration_chat.extract_lake_memory_from_assistant(str(m.get("content") or ""))
            if mem:
                self._lake_language_memory = mem
                break
        self._history_id = str(data.get("id") or hid)
        self._history_title = str(data.get("title") or self._history_id)
        saved_system = str(data.get("system_prompt") or "").strip()
        if saved_system:
            self._apply_traditional_chat_mode(False, notify=False)
            self.system_prompt_edit.setPlainText(saved_system)
            self._config.system_prompt = saved_system
            self._refresh_system_lang_btn()
        else:
            self._apply_traditional_chat_mode(True, notify=False)
        self._on_clear_chat_image_clicked(notify=False)
        self._render_messages_to_view()
        self._refresh_history_title_label()
        self._suppress_send_until = _eai_platform_bootstrap.time.monotonic() + 0.4
        self.input_edit.setPlainText(draft)
        cursor = self.input_edit.textCursor()
        cursor.movePosition(cursor.End)
        self.input_edit.setTextCursor(cursor)
        self._focus_chat_input()
        tip = "已放入输入框，点「发送」后再请求" if draft else "已加载（无用户消息可编辑）"
        self._append_system_line(f"已加载历史对话: {self._history_title} — {tip}")
        self.status_message.emit(f"已加载历史对话: {self._history_title}（待发送）")

    def _clear_chat(self) -> None:
        self._messages = []
        self._lake_language_memory = _eai_configuration_chat.lake_default_previous_subtask(
            _eai_configuration_chat.lake_orchestrator_prompt_lang(self._config.system_prompt)
        )
        self.history_view.clear()
        self._history_id = ""
        self._history_title = ""
        self._refresh_history_title_label()
        self._on_clear_chat_image_clicked(notify=False)
        self._append_system_line("对话已清空")

    def _build_api_messages(self, user_text: str) -> _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]]:
        """历史保持纯文本；当前轮可附带本地选图或相机 JPEG（vision）。

        Lake / 本地远程 Qwen：对齐训练格式
          system + user(任务 / 输出要求) + image
        """
        use_lake = (
            not self._traditional_chat_mode
            and _eai_configuration_chat.should_use_lake_orchestrator_prompt(
                self._config.api_base, self._config.model
            )
        )
        prompt_text = (
            _eai_configuration_chat.format_lake_user_prompt(
                user_text,
                memory=self._lake_language_memory,
                system_prompt=self._config.system_prompt,
            )
            if use_lake
            else user_text
        )
        if use_lake:
            self._append_system_line(
                "已按 Lake 格式包装提示词（system 与 user 输出要求已对齐）"
            )

        api_messages: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
        system_text = self._config.system_prompt.strip()
        if use_lake and system_text:
            api_messages.append({"role": "system", "content": system_text})
        elif not use_lake:
            api_messages = [
                m
                for m in self._messages[:-1]
                if str(m.get("role") or "") in ("user", "assistant")
            ]

        content: object = prompt_text
        image_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        image_tag = ""

        if self._chat_attach_image_bgr is not None:
            image_bgr = self._chat_attach_image_bgr
            image_tag = self._chat_attach_image_path or "本地图片"
        elif (
            self.attach_camera_check.isChecked()
            and self._camera_frame_provider is not None
        ):
            frame = self._camera_frame_provider()
            if frame is not None:
                topic, image_bgr = frame
                image_tag = topic
            else:
                self._append_system_line("未获取到相机帧，仅发送文本")

        if image_bgr is not None:
            try:
                b64 = _eai_adapters_models_chat_client.encode_bgr_image_jpeg_b64(image_bgr)
                # 训练数据为 type=image；OpenAI 兼容接口用 image_url
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt_text},
                ]
                self._append_system_line(f"已附带图片: {image_tag}")
            except Exception as exc:
                self._append_system_line(f"附带图片失败，仅发送文本: {exc}")

        api_messages.append({"role": "user", "content": content})
        return api_messages

    def _on_send_clicked(self) -> None:
        if self._busy:
            return
        if self._current_running_service() is None:
            self._append_error_line(
                "没有可用的运行中推理服务。请先部署启动，再点「探测」刷新列表。"
            )
            self.status_message.emit("无运行中的推理服务")
            return
        # 强制使用当前下拉选中的运行中服务，避免设置面板残留地址
        svc = self._current_running_service()
        if svc is not None:
            self._apply_running_service(svc, silent=True)
        text = self.input_edit.toPlainText().strip()
        if not text:
            return
        if svc and str(svc.get("family") or "") == "Bagel":
            self._send_bagel_chat(text, svc)
            return
        self._sync_config_from_ui()
        self.input_edit.clear()
        send_ts = _eai_adapters_models_chat_client._utc_now_iso()
        self._request_t0 = _eai_platform_bootstrap.time.perf_counter()
        self._append_user_line(text, ts=send_ts)
        self._messages.append({"role": "user", "content": text, "ts": send_ts})
        self._trim_history()
        if not (self._history_title or "").strip():
            self._history_title = _eai_infrastructure_ui_storage.default_chat_history_title(self._messages)
            self._refresh_history_title_label()
        api_messages = self._build_api_messages(text)
        self._append_prompt_dump(api_messages)
        self._set_busy(True)
        self.status_message.emit("正在调用大模型…")

        def _work() -> None:
            try:
                reply = self._client.chat(api_messages)
                self._bridge.finished.emit(reply, True)
            except Exception as exc:
                self._bridge.finished.emit(str(exc), False)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def set_extra_inference_candidates(
        self, items: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Sequence[_eai_platform_bootstrap.Tuple[str, str, str, str]]]
    ) -> None:
        self._extra_inference_candidates = list(items or [])

    def _send_bagel_chat(self, text: str, svc: _eai_platform_bootstrap.Dict[str, str]) -> None:
        self.input_edit.clear()
        send_ts = _eai_adapters_models_chat_client._utc_now_iso()
        self._request_t0 = _eai_platform_bootstrap.time.perf_counter()
        self._append_user_line(text, ts=send_ts)
        self._messages.append({"role": "user", "content": text, "ts": send_ts})
        self._trim_history()
        if not (self._history_title or "").strip():
            self._history_title = _eai_infrastructure_ui_storage.default_chat_history_title(self._messages)
            self._refresh_history_title_label()
        image_bgr = None
        if self._chat_attach_image_bgr is not None:
            image_bgr = self._chat_attach_image_bgr
        elif (
            self.attach_camera_check.isChecked()
            and self._camera_frame_provider is not None
        ):
            frame = self._camera_frame_provider()
            if frame is not None:
                image_bgr = frame[1]
        api_base = str(svc.get("api_base") or "").rstrip("/")
        self._append_system_line(
            f"Bagel 调用: {'图像理解' if image_bgr is not None else '文生图'}  {api_base}"
        )
        self._set_busy(True)
        self.status_message.emit("正在调用 Bagel…")
        img_copy = None if image_bgr is None else _eai_platform_bootstrap.np.asarray(image_bgr).copy()

        def _work() -> None:
            try:
                reply = _eai_adapters_models_chat_client.call_bagel_inference(api_base, text, image_bgr=img_copy)[0]
                self._bridge.finished.emit(reply, True)
            except Exception as exc:
                self._bridge.finished.emit(str(exc), False)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_llm_finished(self, text: str, ok: bool) -> None:
        self._set_busy(False)
        recv_ts = _eai_adapters_models_chat_client._utc_now_iso()
        latency_s: _eai_platform_bootstrap.Optional[float] = None
        if self._request_t0 is not None:
            latency_s = max(0.0, _eai_platform_bootstrap.time.perf_counter() - self._request_t0)
        self._request_t0 = None
        if ok:
            self._append_assistant_line(text, ts=recv_ts, latency_s=latency_s)
            mem = _eai_configuration_chat.extract_lake_memory_from_assistant(text)
            if mem:
                self._lake_language_memory = mem
            msg: _eai_platform_bootstrap.Dict[str, object] = {
                "role": "assistant",
                "content": text,
                "ts": recv_ts,
            }
            if latency_s is not None:
                msg["latency_s"] = float(latency_s)
            self._messages.append(msg)
            self._trim_history()
            lat = _eai_infrastructure_ui_storage.format_chat_latency(latency_s)
            self.status_message.emit(
                f"大模型回复完成" + (f"（耗时 {lat}）" if lat else "")
            )
        else:
            self._append_error_line(text, ts=recv_ts, latency_s=latency_s)
            if self._messages and self._messages[-1].get("role") == "user":
                self._messages.pop()
            lat = _eai_infrastructure_ui_storage.format_chat_latency(latency_s)
            self.status_message.emit(
                f"大模型调用失败" + (f"（耗时 {lat}）: " if lat else ": ") + text[:80]
            )

