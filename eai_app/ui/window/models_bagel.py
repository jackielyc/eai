"""ui.window.models bagel — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_bagel_runtime = module_ref('eai_app.adapters.models.bagel_runtime')
_eai_adapters_models_chat_client = module_ref('eai_app.adapters.models.chat_client')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_chat_history_dialogs = module_ref('eai_app.ui.chat.history_dialogs')

class ModelsBagelMixin:
    def _on_bagel_status(self, text: str) -> None:
        self.bagel_status_label.setText(text if text.startswith("Bagel") else f"Bagel: {text}")
        self.status_bar.showMessage(text)

    def _on_bagel_root_browse_clicked(self) -> None:
        current = self.bagel_root_edit.text().strip() or _eai_configuration_models.BAGEL_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 Bagel 仓库目录", current
        )
        if selected:
            self.bagel_root_edit.setText(selected)
            py = _eai_adapters_models_bagel_runtime.resolve_bagel_python(_eai_adapters_models_bagel_runtime.resolve_bagel_root(selected))
            self.bagel_python_edit.setText(py)

    def _on_bagel_python_browse_clicked(self) -> None:
        current = self.bagel_python_edit.text().strip() or _eai_platform_bootstrap.sys.executable
        initial = _eai_platform_bootstrap.os.path.dirname(current) if current else ""
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self, "选择 Python 解释器", initial, "Python (python*);;All (*)"
        )
        if selected:
            self.bagel_python_edit.setText(selected)

    def _on_bagel_model_browse_clicked(self) -> None:
        repo = _eai_adapters_models_bagel_runtime.resolve_bagel_root(self.bagel_root_edit.text())
        current = self.bagel_model_edit.text().strip() or _eai_configuration_models.BAGEL_MODEL_DEFAULT
        initial = current if _eai_platform_bootstrap.os.path.isabs(current) else _eai_platform_bootstrap.os.path.join(repo, current)
        if not _eai_platform_bootstrap.os.path.isdir(initial):
            initial = repo
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 BAGEL 模型目录", initial)
        if selected:
            # 尽量写成相对仓库路径
            try:
                rel = _eai_platform_bootstrap.os.path.relpath(selected, repo)
                if not rel.startswith(".."):
                    self.bagel_model_edit.setText(rel)
                else:
                    self.bagel_model_edit.setText(selected)
            except ValueError:
                self.bagel_model_edit.setText(selected)

    def _on_bagel_download_clicked(self) -> None:
        save_dir = _eai_configuration_models.BAGEL_MODEL_CACHE_DEFAULT
        py = self.bagel_python_edit.text().strip() or _eai_adapters_models_bagel_runtime.resolve_bagel_python(
            _eai_adapters_models_bagel_runtime.resolve_bagel_root(self.bagel_root_edit.text())
        )
        if not _eai_platform_bootstrap.os.path.isfile(py):
            _eai_platform_bootstrap.QMessageBox.warning(self, "Bagel", f"Python 不存在: {py}")
            return
        _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(save_dir), exist_ok=True)
        script = (
            "import os\n"
            "from huggingface_hub import snapshot_download\n"
            f"save_dir = {save_dir!r}\n"
            "os.makedirs(save_dir, exist_ok=True)\n"
            "print('HF_ENDPOINT=', os.environ.get('HF_ENDPOINT', ''))\n"
            "print('downloading to', save_dir, flush=True)\n"
            "path = snapshot_download(\n"
            "  repo_id='ByteDance-Seed/BAGEL-7B-MoT',\n"
            "  local_dir=save_dir,\n"
            "  resume_download=True,\n"
            "  allow_patterns=['*.json','*.safetensors','*.bin','*.py','*.md','*.txt'],\n"
            ")\n"
            "print('done', path, flush=True)\n"
        )
        cmd = (
            "unset PYTHONPATH PYTHONHOME || true && "
            "export PYTHONNOUSERSITE=1 "
            "HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} "
            "PYTHONUNBUFFERED=1 && "
            f"exec {_eai_platform_bootstrap.shlex.quote(py)} -c {_eai_platform_bootstrap.shlex.quote(script)}"
        )
        if getattr(self, "_bagel_download_proc", None) is not None:
            proc = self._bagel_download_proc
            if proc.state() == _eai_platform_bootstrap.QProcess.Running:
                self.status_bar.showMessage("模型下载已在进行…")
                return
        proc = _eai_platform_bootstrap.QProcess(self)
        proc.setProcessChannelMode(_eai_platform_bootstrap.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self._append_bagel_log(
                bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").rstrip()
            )
        )

        def _on_finished(code: int, _status: _eai_platform_bootstrap.QProcess.ExitStatus) -> None:
            if code == 0 and _eai_platform_bootstrap.os.path.isdir(save_dir):
                self.bagel_model_edit.setText(save_dir)
                self._append_bagel_log(f"模型已就绪: {save_dir}")
                self.status_bar.showMessage(f"Bagel 模型已下载: {save_dir}")
            else:
                self.status_bar.showMessage(f"Bagel 模型下载失败 (code={code})")
            self.bagel_download_btn.setEnabled(True)

        proc.finished.connect(_on_finished)
        self._bagel_download_proc = proc
        self.bagel_download_btn.setEnabled(False)
        self._append_bagel_log(f"$ 下载模型 → {save_dir}")
        proc.setProcessEnvironment(_eai_adapters_models_bagel_runtime.bagel_isolated_process_env(py))
        proc.start("bash", ["-lc", cmd])

    def _on_bagel_start_clicked(self) -> None:
        host = self.bagel_host_edit.text().strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        port = int(self.bagel_port_spin.value())
        # 已部署 Gradio：直接复用，无需因 API 占用而拦截
        if _eai_adapters_models_bagel_runtime.bagel_probe_gradio_ready(host, port):
            self._bagel_launcher.start(
                repo_dir=self.bagel_root_edit.text(),
                model_path=self.bagel_model_edit.text(),
                python_bin=self.bagel_python_edit.text(),
                server_name=host,
                server_port=port,
                mode=int(self.bagel_mode_combo.currentData() or 1),
                zh=self.bagel_zh_check.isChecked(),
                share=self.bagel_share_check.isChecked(),
            )
            self._update_bagel_run_ui()
            return
        if getattr(self, "_bagel_api_launcher", None) is not None and self._bagel_api_launcher.is_running():
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "Bagel",
                "推理 API 仍在运行，会与 Gradio 争用 GPU。请先「停止 API」。",
            )
            return
        self._bagel_launcher.start(
            repo_dir=self.bagel_root_edit.text(),
            model_path=self.bagel_model_edit.text(),
            python_bin=self.bagel_python_edit.text(),
            server_name=host,
            server_port=port,
            mode=int(self.bagel_mode_combo.currentData() or 1),
            zh=self.bagel_zh_check.isChecked(),
            share=self.bagel_share_check.isChecked(),
        )
        self._update_bagel_run_ui()

    def _on_bagel_stop_clicked(self) -> None:
        # Gradio「停止」也清掉推理 API 等全部相关进程
        if getattr(self, "_bagel_api_launcher", None) is not None:
            try:
                self._bagel_api_launcher._clear_adopted(emit_stopped=False)
                self._bagel_api_launcher._process = None
                self._bagel_api_launcher._ready_emitted = False
                self._bagel_api_launcher.running_changed.emit(False)
            except Exception:
                pass
        self._bagel_launcher.stop()
        self._update_bagel_run_ui()
        self._update_bagel_api_ui()

    def _on_bagel_clear_log_clicked(self) -> None:
        self.bagel_log_edit.clear()

    def _on_bagel_open_clicked(self) -> None:
        url = self._bagel_launcher.current_url() or _eai_adapters_models_bagel_runtime.bagel_app_url(
            self.bagel_host_edit.text(), int(self.bagel_port_spin.value())
        )
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl(url))
        self.status_bar.showMessage(f"已打开 {url}")

    def _on_bagel_ready_url(self, url: str) -> None:
        self._load_bagel_web(url)
        self.bagel_open_btn.setEnabled(True)
        if hasattr(self, "bagel_pick_image_btn"):
            self.bagel_pick_image_btn.setEnabled(self.bagel_web_view is not None)

    def _on_bagel_api_status(self, text: str) -> None:
        label = text if text.startswith("Bagel") else f"推理 API: {text}"
        self.bagel_api_status_label.setText(label)
        self.status_bar.showMessage(text)

    def _on_bagel_api_start_clicked(self) -> None:
        host = self.bagel_host_edit.text().strip() or _eai_configuration_models.BAGEL_SERVER_HOST_DEFAULT
        port = int(self.bagel_api_port_spin.value())
        # 先探测已部署实例；命中则直接复用，无需停 Gradio
        if _eai_adapters_models_bagel_runtime.bagel_probe_api_health(host, port) is not None:
            self._bagel_api_launcher.start(
                repo_dir=self.bagel_root_edit.text(),
                model_path=self.bagel_model_edit.text(),
                python_bin=self.bagel_python_edit.text(),
                server_name=host,
                server_port=port,
                mode=int(self.bagel_mode_combo.currentData() or 1),
            )
            self._update_bagel_api_ui()
            return
        if self._bagel_launcher.is_running():
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "Bagel",
                "Gradio 仍在运行，会与推理 API 争用同一份 GPU 权重。\n"
                "请先停止 Gradio，再部署推理 API。",
            )
            return
        self._bagel_api_launcher.start(
            repo_dir=self.bagel_root_edit.text(),
            model_path=self.bagel_model_edit.text(),
            python_bin=self.bagel_python_edit.text(),
            server_name=host,
            server_port=port,
            mode=int(self.bagel_mode_combo.currentData() or 1),
        )
        self._update_bagel_api_ui()

    def _on_bagel_api_stop_clicked(self) -> None:
        # 同时清掉 Gradio launcher 状态（进程会被 pkill 一并杀掉）
        if getattr(self, "_bagel_launcher", None) is not None:
            try:
                self._bagel_launcher._process = None
                self._bagel_launcher._ready_emitted = False
                self._bagel_launcher.running_changed.emit(False)
            except Exception:
                pass
        self._bagel_api_launcher.stop()
        self._update_bagel_api_ui()
        self._update_bagel_run_ui()

    def _on_bagel_api_ready(self, url: str) -> None:
        mode_txt = _eai_adapters_models_bagel_runtime.bagel_mode_label(self._bagel_api_launcher.current_mode())
        # 若服务 /health 已回传 mode，以其为准
        try:
            ok, body, _err = _eai_adapters_models_chat_client._http_get_json(url.rstrip("/") + "/health", timeout_s=1.5)
            if ok and isinstance(body, dict) and body.get("mode") is not None:
                mode_txt = _eai_adapters_models_bagel_runtime.bagel_mode_label(int(body.get("mode") or 1))
                self._bagel_api_launcher._mode = int(body.get("mode") or 1)
        except Exception:
            pass
        self.bagel_api_status_label.setText(
            f"推理 API: {'已复用' if getattr(self._bagel_api_launcher, '_adopted', False) else '已就绪'} "
            f"{url}  ·  mode={mode_txt}"
        )
        self.bagel_api_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        self.bagel_call_btn.setEnabled(not self._bagel_call_busy)
        api_base = url.rstrip("/") + "/v1"
        self.chat_panel.set_extra_inference_candidates(
            [("Bagel", api_base, "EMPTY", "BAGEL-7B-MoT")]
        )
        n = self.chat_panel.refresh_running_providers(
            prefer_api=api_base,
            prefer_model="BAGEL-7B-MoT",
            notify=True,
            silent_apply=False,
        )
        self._append_bagel_log(
            f"[ready] 推理 API {url}  mode={mode_txt}  已加入对话服务列表 ({n})"
        )

    def _on_bagel_image_size_changed(self, *_args: object) -> None:
        """尺寸预设变化时同步 W/H 显示；W/H 始终可调。"""
        if not hasattr(self, "bagel_image_w_spin"):
            return
        ratio = str(self.bagel_image_size_combo.currentData() or "")
        if ratio.isdigit():
            side = max(16, int(ratio))
            self._bagel_wh_syncing = True
            try:
                self.bagel_image_w_spin.setValue(side)
                self.bagel_image_h_spin.setValue(side)
            finally:
                self._bagel_wh_syncing = False
        self.bagel_image_w_spin.setEnabled(True)
        self.bagel_image_h_spin.setEnabled(True)

    def _on_bagel_image_wh_changed(self, *_args: object) -> None:
        """用户改 W/H 时自动切到「手动 W×H」。"""
        if getattr(self, "_bagel_wh_syncing", False):
            return
        idx = self.bagel_image_size_combo.findData("custom")
        if idx < 0:
            return
        if self.bagel_image_size_combo.currentIndex() != idx:
            self.bagel_image_size_combo.blockSignals(True)
            self.bagel_image_size_combo.setCurrentIndex(idx)
            self.bagel_image_size_combo.blockSignals(False)

    def _bagel_selected_image_size(self) -> _eai_platform_bootstrap.Tuple[str, int, int]:
        """返回 (image_ratio, width, height)；手动模式时 width/height > 0。"""
        ratio = str(
            self.bagel_image_size_combo.currentData() or _eai_configuration_models.BAGEL_IMAGE_SIZE_DEFAULT
        )
        width = 0
        height = 0
        if ratio == "custom":
            width = int(self.bagel_image_w_spin.value())
            height = int(self.bagel_image_h_spin.value())
        return ratio, width, height

    def _on_bagel_call_pick_clicked(self) -> None:
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 Bagel 输入图",
            _eai_platform_bootstrap.os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All (*)",
        )
        if not selected:
            return
        self._set_bagel_call_input_image(selected)

    def _on_bagel_import_from_sub_task_clicked(self) -> None:
        """把 sub task 选中图 + AI 对话最近输出导入为 Bagel 调用输入。"""
        image_path = ""
        image_title = ""
        scenario = str(getattr(self, "_test_selected_scenario", "") or "").strip()
        if scenario:
            image_path = str(self._test_image_paths.get(scenario) or "").strip()
            image_title = scenario
        if not image_path:
            image_path = self.chat_panel.get_attached_image_path()
            if image_path:
                image_title = _eai_platform_bootstrap.os.path.basename(image_path)

        out_text = self.chat_panel.get_last_role_text("assistant")
        if not out_text:
            out_text = self.chat_panel.get_last_role_text("user")

        if not image_path and not out_text:
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "sub image",
                "sub task 中尚无可导入内容。\n"
                "请先在 sub task 点选场景图，并在 AI 对话中得到模型输出。",
            )
            return

        imported: _eai_platform_bootstrap.List[str] = []
        if image_path:
            if self._set_bagel_call_input_image(image_path, display_name=image_title):
                imported.append(f"图像: {image_title or _eai_platform_bootstrap.os.path.basename(image_path)}")
            else:
                _eai_platform_bootstrap.QMessageBox.warning(self, "sub image", f"无法读取图像: {image_path}")
                return
        if out_text:
            self.bagel_call_prompt_edit.setPlainText(out_text)
            preview = out_text.replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "…"
            imported.append(f"提示词: {preview}")

        # 有图时默认走编辑；仅文本则文生图
        if image_path:
            idx = self.bagel_call_task_combo.findData("edit")
            if idx >= 0:
                self.bagel_call_task_combo.setCurrentIndex(idx)
        else:
            idx = self.bagel_call_task_combo.findData("text2image")
            if idx >= 0:
                self.bagel_call_task_combo.setCurrentIndex(idx)

        msg = "已从 sub task 导入: " + "；".join(imported)
        self._append_bagel_log(f"[import] {msg}")
        self.status_bar.showMessage(msg)

    def _on_bagel_fast_preset_clicked(self) -> None:
        """一键加速：小分辨率 + 少步数 + cfg_interval + TaylorSeer。"""
        self.bagel_num_timesteps_spin.setValue(_eai_configuration_models.BAGEL_FAST_NUM_TIMESTEPS)
        idx = self.bagel_image_size_combo.findData(_eai_configuration_models.BAGEL_FAST_IMAGE_SIZE)
        if idx >= 0:
            self.bagel_image_size_combo.setCurrentIndex(idx)
        if hasattr(self, "bagel_cfg_interval_spin"):
            self.bagel_cfg_interval_spin.setValue(_eai_configuration_models.BAGEL_CFG_INTERVAL_DEFAULT)
        if hasattr(self, "bagel_taylorseer_check"):
            self.bagel_taylorseer_check.setChecked(True)
        self._append_bagel_log(
            f"[preset] 快速: steps={_eai_configuration_models.BAGEL_FAST_NUM_TIMESTEPS}  "
            f"size={_eai_configuration_models.BAGEL_FAST_IMAGE_SIZE}  "
            f"cfg_interval={_eai_configuration_models.BAGEL_CFG_INTERVAL_DEFAULT}  TaylorSeer=on"
        )
        self.status_bar.showMessage(
            f"已应用快速预设: {_eai_configuration_models.BAGEL_FAST_IMAGE_SIZE}² · steps={_eai_configuration_models.BAGEL_FAST_NUM_TIMESTEPS}"
        )

    def _on_bagel_call_clicked(self) -> None:
        if self._bagel_call_busy:
            return
        if not self._bagel_api_launcher.is_running():
            _eai_platform_bootstrap.QMessageBox.information(self, "Bagel", "请先点「部署推理 API」并等待就绪。")
            return
        prompt = self.bagel_call_prompt_edit.toPlainText().strip()
        if not prompt:
            _eai_platform_bootstrap.QMessageBox.information(self, "Bagel", "请填写提示词。")
            return
        task = str(self.bagel_call_task_combo.currentData() or "text2image")
        image_bgr = None
        if task in ("understand", "edit"):
            path = self._bagel_call_image_path
            if not path or not _eai_platform_bootstrap.os.path.isfile(path):
                _eai_platform_bootstrap.QMessageBox.information(self, "Bagel", "理解/编辑请先点「选图」。")
                return
            image_bgr = _eai_platform_bootstrap.cv2.imread(path, _eai_platform_bootstrap.cv2.IMREAD_COLOR)
            if image_bgr is None:
                _eai_platform_bootstrap.QMessageBox.warning(self, "Bagel", f"无法读取图片: {path}")
                return
        api_base = self._bagel_api_launcher.api_base()
        steps = int(self.bagel_num_timesteps_spin.value())
        image_ratio, image_width, image_height = self._bagel_selected_image_size()
        cfg_interval = (
            float(self.bagel_cfg_interval_spin.value())
            if hasattr(self, "bagel_cfg_interval_spin")
            else _eai_configuration_models.BAGEL_CFG_INTERVAL_DEFAULT
        )
        enable_taylorseer = bool(
            hasattr(self, "bagel_taylorseer_check")
            and self.bagel_taylorseer_check.isChecked()
        )
        size_txt = (
            f"{image_width}x{image_height}"
            if image_ratio == "custom" and image_width > 0 and image_height > 0
            else image_ratio
        )
        self._bagel_call_busy = True
        self.bagel_call_btn.setEnabled(False)
        self.bagel_call_btn.setText("调用中…")
        self._append_bagel_log(
            f"[call] {task}: size={size_txt}  steps={steps}  "
            f"cfg_interval={cfg_interval}  taylor={int(enable_taylorseer)}  "
            f"{prompt[:80]}"
        )
        img_copy = None if image_bgr is None else _eai_platform_bootstrap.np.asarray(image_bgr).copy()
        input_image_path = str(self._bagel_call_image_path or "")

        def _work() -> None:
            try:
                reply, latency_s, meta = _eai_adapters_models_chat_client.call_bagel_inference(
                    api_base,
                    prompt,
                    image_bgr=img_copy,
                    task=task,
                    num_timesteps=steps,
                    image_ratio=image_ratio,
                    image_width=image_width,
                    image_height=image_height,
                    cfg_interval=cfg_interval,
                    enable_taylorseer=enable_taylorseer,
                )
                payload_out: _eai_platform_bootstrap.Dict[str, object] = {
                    "ok": True,
                    "reply": reply,
                    "latency_s": latency_s,
                    "task": task,
                    "prompt": prompt,
                    "image_ratio": image_ratio,
                    "image_width": image_width,
                    "image_height": image_height,
                    "num_timesteps": steps,
                    "cfg_interval": cfg_interval,
                    "enable_taylorseer": enable_taylorseer,
                    "input_image": input_image_path,
                }
                payload_out.update(meta)
                self._bagel_call_bridge.finished.emit(payload_out)
            except Exception as exc:
                self._bagel_call_bridge.finished.emit(
                    {
                        "ok": False,
                        "error": str(exc),
                        "task": task,
                        "prompt": prompt,
                        "image_ratio": image_ratio,
                        "image_width": image_width,
                        "image_height": image_height,
                        "num_timesteps": steps,
                        "cfg_interval": cfg_interval,
                        "enable_taylorseer": enable_taylorseer,
                        "input_image": input_image_path,
                    }
                )

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_bagel_call_finished(self, result: object) -> None:
        self._bagel_call_busy = False
        self.bagel_call_btn.setText("调用")
        self.bagel_call_btn.setEnabled(self._bagel_api_launcher.is_running())
        data = result if isinstance(result, dict) else {}
        latency_s = data.get("latency_s")
        try:
            latency_txt = f"{float(latency_s):.2f}s"
        except (TypeError, ValueError):
            latency_txt = "—"
        if hasattr(self, "bagel_call_latency_label"):
            self.bagel_call_latency_label.setText(f"调用耗时: {latency_txt}")
        model_w = data.get("model_input_width")
        model_h = data.get("model_input_height")
        try:
            model_txt = f"{int(model_w)}×{int(model_h)}"
        except (TypeError, ValueError):
            model_txt = "—"
        if hasattr(self, "bagel_call_model_size_label"):
            self.bagel_call_model_size_label.setText(f"模型输入: {model_txt}")
        if not data.get("ok"):
            err = str(data.get("error") or result)
            self._append_bagel_log(f"[call error] {err}")
            self.bagel_call_output_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
            self.bagel_call_output_preview.setText("调用失败")
            if hasattr(self, "bagel_call_output_size_label"):
                self.bagel_call_output_size_label.setText("输出分辨率: —")
            if hasattr(self, "bagel_call_model_size_label"):
                self.bagel_call_model_size_label.setText("模型输入: —")
            self.status_bar.showMessage(f"Bagel 调用失败: {err[:80]}")
            try:
                self._bagel_history_append_turn(
                    task=str(data.get("task") or ""),
                    prompt=str(data.get("prompt") or ""),
                    image_ratio=str(data.get("image_ratio") or ""),
                    image_width=int(data.get("image_width") or 0),
                    image_height=int(data.get("image_height") or 0),
                    num_timesteps=int(data.get("num_timesteps") or 0),
                    input_image=str(
                        data.get("input_image") or self._bagel_call_image_path or ""
                    ),
                    output_image="",
                    output_text="",
                    latency_s=None,
                    ok=False,
                    error=err,
                    autosave=True,
                )
            except Exception as exc:
                self._append_bagel_log(f"[history warn] 保存失败轮次失败: {exc}")
            return
        reply = str(data.get("reply") or "")
        path = ""
        text = reply
        if reply.startswith("BAGEL_IMAGE::"):
            first, _, rest = reply.partition("\n")
            path = first.split("::", 1)[-1].strip()
            text = rest
        if path and _eai_platform_bootstrap.os.path.isfile(path):
            pix = _eai_platform_bootstrap.QPixmap(path)
            if not pix.isNull():
                self.bagel_call_output_preview.setPixmap(
                    pix.scaled(
                        self.bagel_call_output_preview.size(),
                        _eai_platform_bootstrap.Qt.KeepAspectRatio,
                        _eai_platform_bootstrap.Qt.SmoothTransformation,
                    )
                )
                self.bagel_call_output_preview.setText("")
                self.bagel_call_output_preview.setToolTip(path)
                if hasattr(self, "bagel_call_output_size_label"):
                    self.bagel_call_output_size_label.setText(
                        f"输出分辨率: {pix.width()}×{pix.height()}"
                    )
            elif hasattr(self, "bagel_call_output_size_label"):
                self.bagel_call_output_size_label.setText("输出分辨率: —")
            self._append_bagel_log(
                f"[call ok] {latency_txt}  模型输入:{model_txt}  图像: {path}"
            )
        else:
            self.bagel_call_output_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
            preview = text.strip() or "(无图像输出)"
            self.bagel_call_output_preview.setText(preview[:400])
            if hasattr(self, "bagel_call_output_size_label"):
                self.bagel_call_output_size_label.setText("输出分辨率: —")
            self._append_bagel_log(
                f"[call ok] {latency_txt}  模型输入:{model_txt}  {preview[:200]}"
            )
        self.status_bar.showMessage(f"Bagel 调用完成 · {latency_txt}")
        # 写入当前会话并自动保存
        try:
            lat_val: _eai_platform_bootstrap.Optional[float] = None
            try:
                lat_val = float(latency_s) if latency_s is not None else None
            except (TypeError, ValueError):
                lat_val = None
            self._bagel_history_append_turn(
                task=str(data.get("task") or ""),
                prompt=str(data.get("prompt") or ""),
                image_ratio=str(data.get("image_ratio") or ""),
                image_width=int(data.get("image_width") or 0),
                image_height=int(data.get("image_height") or 0),
                num_timesteps=int(data.get("num_timesteps") or 0),
                input_image=str(data.get("input_image") or self._bagel_call_image_path or ""),
                output_image=path,
                output_text=text.strip(),
                latency_s=lat_val,
                ok=True,
                error="",
                autosave=True,
            )
        except Exception as exc:
            self._append_bagel_log(f"[history warn] 保存轮次失败: {exc}")

    def _on_bagel_history_new_clicked(self) -> None:
        self._bagel_history_id = ""
        self._bagel_history_title = ""
        self._bagel_history_turns = []
        self._bagel_call_image_path = ""
        self.bagel_call_input_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
        self.bagel_call_input_preview.setText("输入图")
        self.bagel_call_input_preview.setToolTip("")
        if hasattr(self, "bagel_call_input_size_label"):
            self.bagel_call_input_size_label.setText("输入分辨率: —")
        self.bagel_call_output_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
        self.bagel_call_output_preview.setText("输出图")
        self.bagel_call_output_preview.setToolTip("")
        if hasattr(self, "bagel_call_output_size_label"):
            self.bagel_call_output_size_label.setText("输出分辨率: —")
        if hasattr(self, "bagel_call_model_size_label"):
            self.bagel_call_model_size_label.setText("模型输入: —")
        if hasattr(self, "bagel_call_latency_label"):
            self.bagel_call_latency_label.setText("调用耗时: —")
        self._refresh_bagel_history_title_label()
        self._refresh_bagel_history_turns_list()
        self._append_bagel_log("[history] 已新建空对话")
        self.status_bar.showMessage("Bagel: 新对话")

    def _on_bagel_history_save_clicked(self) -> None:
        if not self._bagel_history_turns and not self._bagel_history_id:
            _eai_platform_bootstrap.QMessageBox.information(self, "Bagel 历史", "当前没有可保存的调用轮次。")
            return
        try:
            path = self._bagel_history_save(ask_title=True)
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "Bagel 历史", f"保存失败: {exc}")
            return
        self._append_bagel_log(f"[history] 已保存: {path}")
        self.status_bar.showMessage(f"Bagel 历史已保存: {self._bagel_history_title}")

    def _on_bagel_history_clicked(self) -> None:
        if self._bagel_history_dialog is not None:
            try:
                self._bagel_history_dialog.close()
            except Exception:
                pass
            self._bagel_history_dialog = None
        dlg = _eai_ui_chat_history_dialogs.BagelHistoryDialog(self)
        self._bagel_history_dialog = dlg

        def _on_finished(code: int) -> None:
            accepted = int(code) == int(_eai_platform_bootstrap.QDialog.Accepted)
            hid = str(getattr(dlg, "selected_id", "") or "")
            self._bagel_history_dialog = None
            if accepted and hid:
                try:
                    self._bagel_history_load_by_id(hid)
                except Exception as exc:
                    _eai_platform_bootstrap.QMessageBox.warning(self, "Bagel 历史", f"加载失败: {exc}")

        dlg.finished.connect(_on_finished)
        dlg.show()
        dlg.raise_()

    def _on_bagel_history_delete_turn_clicked(self) -> None:
        lw = getattr(self, "bagel_history_turns_list", None)
        if lw is None:
            return
        item = lw.currentItem()
        if item is None:
            _eai_platform_bootstrap.QMessageBox.information(self, "Bagel 历史", "请先在左侧选中一轮。")
            return
        idx = int(item.data(_eai_platform_bootstrap.Qt.UserRole) or -1)
        if idx < 0 or idx >= len(self._bagel_history_turns):
            return
        del self._bagel_history_turns[idx]
        if self._bagel_history_id:
            try:
                self._bagel_history_save(ask_title=False)
            except Exception as exc:
                self._append_bagel_log(f"[history warn] 删轮后保存失败: {exc}")
        self._refresh_bagel_history_turns_list()
        self._refresh_bagel_history_title_label()
        self._append_bagel_log(f"[history] 已删除第 {idx + 1} 轮")

    def _on_bagel_history_turn_clicked(self, item: object) -> None:
        if not isinstance(item, _eai_platform_bootstrap.QListWidgetItem):
            return
        idx = int(item.data(_eai_platform_bootstrap.Qt.UserRole) or -1)
        if idx < 0 or idx >= len(self._bagel_history_turns):
            return
        self._bagel_history_apply_turn(self._bagel_history_turns[idx])

    def _bagel_history_save(self, *, ask_title: bool = False) -> str:
        hid = self._bagel_history_id or _eai_platform_bootstrap.uuid.uuid4().hex[:12]
        title = self._bagel_history_title
        if ask_title or not title:
            default = title or _eai_infrastructure_ui_storage.default_bagel_history_title(self._bagel_history_turns)
            text, ok = _eai_platform_bootstrap.QInputDialog.getText(
                self,
                "Bagel 历史标题",
                "会话标题:",
                text=default,
            )
            if ok:
                title = str(text or "").strip() or default
            elif not title:
                title = default
        # 持久化媒体
        turns_out: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Dict[str, object]] = []
        for i, turn in enumerate(self._bagel_history_turns):
            t = dict(turn)
            t["input_image"] = _eai_infrastructure_ui_storage.bagel_persist_media(
                hid, str(t.get("input_image") or ""), f"in{i}"
            ) or str(t.get("input_image") or "")
            t["output_image"] = _eai_infrastructure_ui_storage.bagel_persist_media(
                hid, str(t.get("output_image") or ""), f"out{i}"
            ) or str(t.get("output_image") or "")
            turns_out.append(t)
        self._bagel_history_turns = turns_out
        record: _eai_platform_bootstrap.Dict[str, object] = {
            "id": hid,
            "title": title,
            "turns": turns_out,
        }
        if self._bagel_history_id:
            try:
                old = _eai_infrastructure_ui_storage.load_bagel_history(self._bagel_history_id)
                if old.get("created_at"):
                    record["created_at"] = old["created_at"]
            except Exception:
                pass
        path = _eai_infrastructure_ui_storage.save_bagel_history_record(record)
        self._bagel_history_id = hid
        self._bagel_history_title = title
        self._refresh_bagel_history_title_label()
        return path

    def _bagel_history_load_by_id(self, hid: str) -> None:
        data = _eai_infrastructure_ui_storage.load_bagel_history(hid)
        turns = data.get("turns") if isinstance(data.get("turns"), list) else []
        self._bagel_history_id = str(data.get("id") or hid)
        self._bagel_history_title = str(data.get("title") or self._bagel_history_id)
        self._bagel_history_turns = [dict(t) for t in turns if isinstance(t, dict)]
        self._refresh_bagel_history_title_label()
        self._refresh_bagel_history_turns_list()
        if self._bagel_history_turns:
            last = self._bagel_history_turns[-1]
            self._bagel_history_apply_turn(last)
            lw = getattr(self, "bagel_history_turns_list", None)
            if lw is not None and lw.count() > 0:
                lw.setCurrentRow(lw.count() - 1)
        self._append_bagel_log(
            f"[history] 已加载: {self._bagel_history_title}  "
            f"({len(self._bagel_history_turns)} 轮)"
        )
        self.status_bar.showMessage(f"已加载 Bagel 历史: {self._bagel_history_title}")

    def _bagel_history_apply_turn(self, turn: _eai_platform_bootstrap.Dict[str, object]) -> None:
        task = str(turn.get("task") or "").strip()
        if task:
            idx = self.bagel_call_task_combo.findData(task)
            if idx >= 0:
                self.bagel_call_task_combo.setCurrentIndex(idx)
        prompt = str(turn.get("prompt") or "")
        self.bagel_call_prompt_edit.setPlainText(prompt)
        ratio = str(turn.get("image_ratio") or "").strip()
        if ratio:
            ridx = self.bagel_image_size_combo.findData(ratio)
            if ridx >= 0:
                self.bagel_image_size_combo.setCurrentIndex(ridx)
        try:
            iw = int(turn.get("image_width") or 0)
            ih = int(turn.get("image_height") or 0)
        except (TypeError, ValueError):
            iw, ih = 0, 0
        if ratio == "custom" and iw > 0 and ih > 0:
            if hasattr(self, "bagel_image_w_spin"):
                self.bagel_image_w_spin.setValue(iw)
            if hasattr(self, "bagel_image_h_spin"):
                self.bagel_image_h_spin.setValue(ih)
        self._on_bagel_image_size_changed()
        steps = turn.get("num_timesteps")
        try:
            if steps is not None and int(steps) > 0:
                self.bagel_num_timesteps_spin.setValue(int(steps))
        except (TypeError, ValueError):
            pass
        in_path = str(turn.get("input_image") or "").strip()
        if in_path and _eai_platform_bootstrap.os.path.isfile(in_path):
            self._set_bagel_call_input_image(in_path)
        else:
            self._bagel_call_image_path = ""
            self.bagel_call_input_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
            self.bagel_call_input_preview.setText("输入图")
            if hasattr(self, "bagel_call_input_size_label"):
                self.bagel_call_input_size_label.setText("输入分辨率: —")
        out_path = str(turn.get("output_image") or "").strip()
        out_text = str(turn.get("output_text") or "").strip()
        if out_path and _eai_platform_bootstrap.os.path.isfile(out_path):
            pix = _eai_platform_bootstrap.QPixmap(out_path)
            if not pix.isNull():
                self.bagel_call_output_preview.setPixmap(
                    pix.scaled(
                        self.bagel_call_output_preview.size(),
                        _eai_platform_bootstrap.Qt.KeepAspectRatio,
                        _eai_platform_bootstrap.Qt.SmoothTransformation,
                    )
                )
                self.bagel_call_output_preview.setText("")
                self.bagel_call_output_preview.setToolTip(out_path)
                if hasattr(self, "bagel_call_output_size_label"):
                    self.bagel_call_output_size_label.setText(
                        f"输出分辨率: {pix.width()}×{pix.height()}"
                    )
        else:
            self.bagel_call_output_preview.setPixmap(_eai_platform_bootstrap.QPixmap())
            self.bagel_call_output_preview.setText(out_text[:400] if out_text else "输出图")
            if hasattr(self, "bagel_call_output_size_label"):
                self.bagel_call_output_size_label.setText("输出分辨率: —")
        lat = turn.get("latency_s")
        if hasattr(self, "bagel_call_latency_label"):
            try:
                self.bagel_call_latency_label.setText(
                    f"调用耗时: {float(lat):.2f}s" if lat is not None else "调用耗时: —"
                )
            except (TypeError, ValueError):
                self.bagel_call_latency_label.setText("调用耗时: —")

    def _bagel_history_append_turn(
        self,
        *,
        task: str,
        prompt: str,
        image_ratio: str,
        num_timesteps: int,
        input_image: str,
        output_image: str,
        output_text: str,
        latency_s: _eai_platform_bootstrap.Optional[float],
        ok: bool,
        error: str,
        autosave: bool = True,
        image_width: int = 0,
        image_height: int = 0,
    ) -> None:
        turn: _eai_platform_bootstrap.Dict[str, object] = {
            "ts": _eai_adapters_models_chat_client._utc_now_iso(),
            "task": task,
            "prompt": prompt,
            "image_ratio": image_ratio,
            "image_width": int(image_width or 0),
            "image_height": int(image_height or 0),
            "num_timesteps": int(num_timesteps or 0),
            "input_image": input_image,
            "output_image": output_image,
            "output_text": output_text,
            "ok": bool(ok),
            "error": error or "",
        }
        if latency_s is not None:
            turn["latency_s"] = float(latency_s)
        self._bagel_history_turns.append(turn)
        self._refresh_bagel_history_turns_list()
        lw = getattr(self, "bagel_history_turns_list", None)
        if lw is not None and lw.count() > 0:
            lw.setCurrentRow(lw.count() - 1)
        if autosave:
            try:
                self._bagel_history_save(ask_title=False)
            except Exception as exc:
                self._append_bagel_log(f"[history warn] 自动保存失败: {exc}")
                self._refresh_bagel_history_title_label()
        else:
            self._refresh_bagel_history_title_label()

    def _on_bagel_pick_image_clicked(self) -> None:
        if self.bagel_web_view is None:
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "Bagel",
                "当前无嵌入页，请用「浏览器打开」后在浏览器里上传。",
            )
            return
        if not self._bagel_launcher.is_running():
            _eai_platform_bootstrap.QMessageBox.warning(self, "Bagel", "请先启动 Bagel")
            return
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择要写入 Gradio 的图片",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif);;All (*)",
        )
        if not path:
            return
        try:
            self._inject_bagel_image(path)
        except Exception as exc:
            _eai_platform_bootstrap.QMessageBox.warning(self, "Bagel", f"注入图片失败: {exc}")
            self._append_bagel_log(f"[error] 注入图片失败: {exc}")

