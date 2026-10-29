"""ui.window.models perception — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_video = module_ref('eai_app.ui.widgets.video')

class ModelsPerceptionMixin:
    def _on_lingbot_status(self, msg: str) -> None:
        if hasattr(self, "lingbot_status_label"):
            self.lingbot_status_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_lingbot_source_changed(self, *_args) -> None:
        source = str(self.lingbot_source_combo.currentData() or "camera")
        file_mode = source == "file"
        self.lingbot_path_edit.setEnabled(file_mode)
        self.lingbot_browse_btn.setEnabled(file_mode)
        if source == "example":
            self.lingbot_path_edit.setText(_eai_configuration_tabs.LINGBOT_VISION_EXAMPLE)
        elif source == "camera":
            self.lingbot_path_edit.setPlaceholderText("运行时从当前彩色相机抓帧")
        self._update_lingbot_ui()

    def _on_lingbot_browse_clicked(self) -> None:
        current = self.lingbot_path_edit.text().strip() or _eai_configuration_tabs.LINGBOT_VISION_EXAMPLE
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_models.EAI_DIR
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            initial,
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if selected:
            self.lingbot_path_edit.setText(selected)

    def _on_lingbot_ckpt_browse_clicked(self) -> None:
        current = self.lingbot_ckpt_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_models.LINGBOT_VISION_ROOT_DEFAULT
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 LingBot-Vision checkpoint",
            initial,
            "PyTorch (*.pt *.pth);;All files (*)",
        )
        if selected:
            self.lingbot_ckpt_edit.setText(selected)

    def _on_lingbot_run_clicked(self) -> None:
        if self._lingbot_launcher.is_running():
            return
        input_path = self._resolve_lingbot_input_image()
        if not input_path:
            return
        self.lingbot_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(input_path))
        self.lingbot_pca_label.set_source_pixmap(None)
        variant = str(self.lingbot_variant_combo.currentData() or "small")
        self._lingbot_launcher.start(
            input_path,
            variant=variant,
            ckpt=self.lingbot_ckpt_edit.text().strip(),
            size=int(self.lingbot_size_spin.value()),
        )
        self._update_lingbot_ui()

    def _on_lingbot_stop_clicked(self) -> None:
        self._append_lingbot_log("--- 用户停止 LingBot-Vision ---")
        self._lingbot_launcher.stop()
        self._update_lingbot_ui()

    def _on_lingbot_result(self, result: dict) -> None:
        if not result.get("ok"):
            return
        pca_path = str(result.get("pca") or "")
        if pca_path and _eai_platform_bootstrap.os.path.isfile(pca_path):
            self.lingbot_pca_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(pca_path))
        input_path = str(result.get("input") or "")
        if input_path and _eai_platform_bootstrap.os.path.isfile(input_path):
            self.lingbot_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(input_path))
        grid = result.get("grid") or []
        self._append_lingbot_log(
            f"完成 variant={result.get('variant')} "
            f"grid={grid} embed_dim={result.get('embed_dim')} "
            f"device={result.get('device')}"
        )

    def _on_lingbot_depth_status(self, msg: str) -> None:
        if hasattr(self, "lingbot_depth_status_label"):
            self.lingbot_depth_status_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_lingbot_depth_source_changed(self, *_args) -> None:
        source = str(self.lingbot_depth_source_combo.currentData() or "camera")
        if source == "example":
            ex = int(self.lingbot_depth_example_spin.value())
            ex_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT, "examples", str(ex))
            self.lingbot_depth_rgb_edit.setPlaceholderText(_eai_platform_bootstrap.os.path.join(ex_dir, "rgb.*"))
            self.lingbot_depth_depth_edit.setPlaceholderText(
                _eai_platform_bootstrap.os.path.join(ex_dir, "raw_depth.png")
            )
            self.lingbot_depth_intrinsics_edit.setPlaceholderText(
                _eai_platform_bootstrap.os.path.join(ex_dir, "intrinsics.txt")
            )
        elif source == "camera":
            self.lingbot_depth_rgb_edit.setPlaceholderText("运行时从当前 RGB-D 抓帧")
            self.lingbot_depth_depth_edit.setPlaceholderText("配对 depth topic")
            self.lingbot_depth_intrinsics_edit.setPlaceholderText("CameraInfo / 默认内参")
        self._update_lingbot_depth_ui()

    def _on_lingbot_depth_model_browse_clicked(self) -> None:
        current = self.lingbot_depth_model_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 LingBot-Depth checkpoint",
            initial,
            "PyTorch (*.pt *.pth);;All files (*)",
        )
        if selected:
            self.lingbot_depth_model_edit.setText(selected)

    def _on_lingbot_depth_rgb_browse_clicked(self) -> None:
        current = self.lingbot_depth_rgb_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 RGB 图像",
            initial,
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if selected:
            self.lingbot_depth_rgb_edit.setText(selected)

    def _on_lingbot_depth_depth_browse_clicked(self) -> None:
        current = self.lingbot_depth_depth_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择原始深度图",
            initial,
            "Depth (*.png *.npy);;All files (*)",
        )
        if selected:
            self.lingbot_depth_depth_edit.setText(selected)

    def _on_lingbot_depth_intrinsics_browse_clicked(self) -> None:
        current = self.lingbot_depth_intrinsics_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_DEPTH_ROOT_DEFAULT
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择内参文件",
            initial,
            "Intrinsics (*.txt *.json);;All files (*)",
        )
        if selected:
            self.lingbot_depth_intrinsics_edit.setText(selected)

    def _on_lingbot_depth_topic_changed(self, *_args) -> None:
        topic = str(self.lingbot_depth_topic_combo.currentData() or "").strip()
        if topic:
            self._last_sam3_topic = topic
        self._update_lingbot_depth_pair_label()

    def _on_lingbot_video_topic_changed(self, *_args) -> None:
        if not hasattr(self, "lingbot_video_topic_combo"):
            return
        topic = str(self.lingbot_video_topic_combo.currentData() or "").strip()
        if topic:
            self._last_sam3_topic = topic

    def _on_lingbot_world_topic_changed(self, *_args) -> None:
        if not hasattr(self, "lingbot_world_topic_combo"):
            return
        topic = str(self.lingbot_world_topic_combo.currentData() or "").strip()
        if topic:
            self._last_sam3_topic = topic

    def _on_lingbot_depth_run_clicked(self) -> None:
        if self._lingbot_depth_launcher.is_running() or getattr(self, "_mv6d_lingbot_active", False):
            self._on_lingbot_depth_status("LingBot-Depth 正在运行，请等待或停止当前任务")
            return
        resolved = self._resolve_lingbot_depth_inputs()
        if resolved is None:
            self._clear_lingbot_depth_result()
            return
        example, rgb, depth, intrinsics = resolved
        self.lingbot_depth_rgb_label.set_source_pixmap(
            _eai_platform_bootstrap.QPixmap(rgb) if rgb and _eai_platform_bootstrap.os.path.isfile(rgb) else None
        )
        self.lingbot_depth_input_label.set_source_pixmap(None)
        self.lingbot_depth_refined_label.set_source_pixmap(None)
        # Bind the result to the captured frame, not a later dropdown selection.
        color_topic = str(getattr(self, "_lingbot_depth_capture_topic", "") or "")
        self._clear_lingbot_depth_result()
        self._lingbot_depth_last_color_topic = color_topic

        model_override = self.lingbot_depth_model_edit.text().strip()
        model = model_override or str(
            self.lingbot_depth_model_combo.currentData() or _eai_configuration_tabs.LINGBOT_DEPTH_MODEL_DEFAULT
        )
        device = str(self.lingbot_depth_device_combo.currentData() or "auto")
        self._lingbot_depth_launcher.start(
            example=example,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            model=model,
            device=device,
            color_topic=color_topic,
        )
        self._update_lingbot_depth_ui()

    def _on_lingbot_depth_stop_clicked(self) -> None:
        self._append_lingbot_depth_log("--- 用户停止 LingBot-Depth ---")
        cancel = getattr(self, "_mv6d_lingbot_cancel", None)
        if getattr(self, "_mv6d_lingbot_active", False) and cancel is not None:
            cancel.set()
            realtime = getattr(self, "mv6d_enable_check", None)
            if realtime is not None:
                realtime.setChecked(False)
            self._on_lingbot_depth_status("已请求停止；当前步骤结束后退出本轮 6D")
            return
        self._lingbot_depth_launcher.stop()
        self._update_lingbot_depth_ui()

    def _on_lingbot_depth_result(self, result: dict) -> None:
        if not result.get("ok"):
            self._clear_lingbot_depth_result()
            return
        for key, label in (
            ("rgb", self.lingbot_depth_rgb_label),
            ("depth_input", self.lingbot_depth_input_label),
            ("depth_refined", self.lingbot_depth_refined_label),
        ):
            path = str(result.get(key) or "")
            if path and _eai_platform_bootstrap.os.path.isfile(path):
                label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
        self._append_lingbot_depth_log(
            f"完成 model={result.get('model')} "
            f"load={result.get('load_s')}s infer={result.get('infer_s')}s "
            f"points={result.get('points')} device={result.get('device')}"
        )
        pixels = int(result.get("width") or 0) * int(result.get("height") or 0)
        if pixels and "input_valid_pixels" in result:
            self._append_lingbot_depth_log(
                f"原始有效深度 {int(result['input_valid_pixels']) / pixels:.1%}，"
                f"精修有效深度 {int(result.get('output_valid_pixels') or 0) / pixels:.1%}；"
                "补全点不等于传感器实测点，显示范围统一为 0.08–2.5m。"
            )
        if result.get("median_change_mm") is not None:
            self._append_lingbot_depth_log(
                f"相对输入深度变化：中位数 {result['median_change_mm']:.1f}mm，"
                f"P95 {result['p95_change_mm']:.1f}mm（非真值误差）。"
            )
        try:
            self._show_lingbot_depth_on_robot_view(result, silent=False)
        except Exception as exc:
            self._append_lingbot_depth_log(f"[WARN] 机器人页未更新: {exc}")

    def _on_lingbot_map_status(self, msg: str) -> None:
        if hasattr(self, "lingbot_map_status_label"):
            self.lingbot_map_status_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_lingbot_map_scene_changed(self, *_args) -> None:
        scene = str(self.lingbot_map_scene_combo.currentData() or "courthouse")
        if scene in ("courthouse", "university", "loop"):
            path = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT, "example", scene)
            self.lingbot_map_path_edit.setText(path)
            self.lingbot_map_mask_sky_check.setChecked(scene != "loop")
            self._lingbot_map_set_preview_from_folder(path)
        elif scene == "camera":
            path = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_MAP_CACHE_DIR, "camera_frames")
            self.lingbot_map_path_edit.setText(path)
            self.lingbot_map_mask_sky_check.setChecked(False)
        self._update_lingbot_map_ui()

    def _lingbot_map_set_preview_from_folder(self, folder: str) -> None:
        if not _eai_platform_bootstrap.os.path.isdir(folder):
            self.lingbot_map_preview_label.set_source_pixmap(None)
            return
        for name in sorted(_eai_platform_bootstrap.os.listdir(folder)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                path = _eai_platform_bootstrap.os.path.join(folder, name)
                self.lingbot_map_preview_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
                return
        self.lingbot_map_preview_label.set_source_pixmap(None)

    def _on_lingbot_map_browse_clicked(self) -> None:
        scene = str(self.lingbot_map_scene_combo.currentData() or "")
        current = self.lingbot_map_path_edit.text().strip() or _eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT
        if scene == "video":
            initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT
            selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
                self,
                "选择视频",
                initial,
                "Video (*.mp4 *.avi *.mov *.mkv);;All files (*)",
            )
            if selected:
                self.lingbot_map_path_edit.setText(selected)
        else:
            initial = current if _eai_platform_bootstrap.os.path.isdir(current) else _eai_configuration_tabs.LINGBOT_MAP_ROOT_DEFAULT
            selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择图像目录", initial)
            if selected:
                self.lingbot_map_path_edit.setText(selected)
                self._lingbot_map_set_preview_from_folder(selected)

    def _on_lingbot_map_model_browse_clicked(self) -> None:
        current = self.lingbot_map_model_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_MAP_MODEL_CACHE
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 LingBot-Map checkpoint",
            initial,
            "PyTorch (*.pt *.pth);;All files (*)",
        )
        if selected:
            self.lingbot_map_model_edit.setText(selected)

    def _lingbot_map_camera_dir(self) -> str:
        path = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_MAP_CACHE_DIR, "camera_frames")
        _eai_platform_bootstrap.os.makedirs(path, exist_ok=True)
        return path

    def _on_lingbot_map_capture_clicked(self) -> None:
        picked = self._pick_sam3_color_source()
        if picked is None:
            self._append_lingbot_map_log("[ERROR] 无可用彩色图像，请勾选 color topic")
            self._on_lingbot_map_status("无可用相机图像")
            return
        topic, image, _panel = picked
        out_dir = self._lingbot_map_camera_dir()
        self._lingbot_map_capture_count = len(
            [
                n
                for n in _eai_platform_bootstrap.os.listdir(out_dir)
                if n.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        idx = self._lingbot_map_capture_count
        path = _eai_platform_bootstrap.os.path.join(out_dir, f"{idx:06d}.png")
        bgr = _eai_platform_bootstrap.np.asarray(image)
        if bgr.ndim == 2:
            bgr = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
        _eai_platform_bootstrap.cv2.imwrite(path, bgr)
        self._lingbot_map_capture_count += 1
        self.lingbot_map_path_edit.setText(out_dir)
        self.lingbot_map_preview_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
        target = int(self.lingbot_map_capture_spin.value())
        self._append_lingbot_map_log(
            f"采集 [{self._lingbot_map_capture_count}/{target}] {topic} -> {path}"
        )
        self._on_lingbot_map_status(
            f"已采集 {self._lingbot_map_capture_count}/{target} 帧"
        )

    def _on_lingbot_map_run_clicked(self) -> None:
        if self._lingbot_map_launcher.is_running():
            return
        resolved = self._resolve_lingbot_map_input()
        if resolved is None:
            return
        image_folder, video_path = resolved
        if image_folder:
            self._lingbot_map_set_preview_from_folder(image_folder)
        model_override = self.lingbot_map_model_edit.text().strip()
        model = model_override or str(
            self.lingbot_map_ckpt_combo.currentData() or _eai_configuration_tabs.LINGBOT_MAP_CKPT_DEFAULT
        )
        self._lingbot_map_viser_url = ""
        self.lingbot_map_url_label.setText("")
        self.lingbot_map_open_btn.setEnabled(False)
        self._lingbot_map_launcher.start(
            image_folder=image_folder,
            video_path=video_path,
            model=model,
            port=int(self.lingbot_map_port_spin.value()),
            first_k=int(self.lingbot_map_first_k_spin.value()),
            stride=int(self.lingbot_map_stride_spin.value()),
            fps=int(self.lingbot_map_fps_spin.value()),
            mask_sky=bool(self.lingbot_map_mask_sky_check.isChecked()),
            use_sdpa=bool(self.lingbot_map_sdpa_check.isChecked()),
        )
        self._update_lingbot_map_ui()

    def _on_lingbot_map_stop_clicked(self) -> None:
        self._append_lingbot_map_log("--- 用户停止 LingBot-Map ---")
        self._lingbot_map_launcher.stop()
        self._update_lingbot_map_ui()

    def _on_lingbot_map_viser_ready(self, url: str) -> None:
        self._lingbot_map_viser_url = url
        self.lingbot_map_url_label.setText(f"viser: {url}  （推理结束后可打开）")
        self.lingbot_map_open_btn.setEnabled(True)
        self._append_lingbot_map_log(f"viser URL: {url}")

    def _on_lingbot_map_open_clicked(self) -> None:
        url = getattr(self, "_lingbot_map_viser_url", "") or (
            f"http://127.0.0.1:{int(self.lingbot_map_port_spin.value())}"
        )
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl(url))

    def _on_lingbot_video_status(self, msg: str) -> None:
        if hasattr(self, "lingbot_video_status_label"):
            self.lingbot_video_status_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_lingbot_video_mode_changed(self, *_args) -> None:
        self._populate_lingbot_video_cases()
        self._on_lingbot_video_preset_changed()
        self._refresh_lingbot_video_topic_combo()
        self._update_lingbot_video_ui()

    def _on_lingbot_video_case_changed(self, *_args) -> None:
        data = self.lingbot_video_case_combo.currentData() or {}
        if not isinstance(data, dict):
            data = {}
        prompt = str(data.get("prompt_json") or "")
        image = str(data.get("image") or "")
        if prompt:
            self.lingbot_video_prompt_edit.setText(prompt)
        if image:
            self.lingbot_video_image_edit.setText(image)
            self.lingbot_video_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(image))
        elif prompt and _eai_platform_bootstrap.os.path.isfile(prompt):
            # no image — clear preview
            self.lingbot_video_input_label.set_source_pixmap(None)
        self._update_lingbot_video_ui()

    def _on_lingbot_video_preset_changed(self, *_args) -> None:
        preset = str(self.lingbot_video_preset_combo.currentData() or "fast")
        mode = str(self.lingbot_video_mode_combo.currentData() or "t2v")
        if preset == "fast":
            self.lingbot_video_h_spin.setValue(192)
            self.lingbot_video_w_spin.setValue(320)
            self.lingbot_video_frames_spin.setValue(9 if mode != "t2i" else 1)
            self.lingbot_video_steps_spin.setValue(20)
        elif preset == "standard":
            self.lingbot_video_h_spin.setValue(480)
            self.lingbot_video_w_spin.setValue(832)
            self.lingbot_video_frames_spin.setValue(121 if mode != "t2i" else 1)
            self.lingbot_video_steps_spin.setValue(40)
        self._update_lingbot_video_ui()

    def _on_lingbot_video_prompt_browse_clicked(self) -> None:
        current = self.lingbot_video_prompt_edit.text().strip()
        initial = (
            current
            if _eai_platform_bootstrap.os.path.isfile(current)
            else _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT, "assets", "cases")
        )
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 prompt.json",
            initial,
            "JSON (*.json);;All files (*)",
        )
        if selected:
            self.lingbot_video_prompt_edit.setText(selected)

    def _on_lingbot_video_image_browse_clicked(self) -> None:
        current = self.lingbot_video_image_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_VIDEO_ROOT_DEFAULT
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 TI2V 首帧",
            initial,
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if selected:
            self.lingbot_video_image_edit.setText(selected)
            self.lingbot_video_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(selected))

    def _on_lingbot_video_model_browse_clicked(self) -> None:
        current = self.lingbot_video_model_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isdir(current) else _eai_configuration_tabs.LINGBOT_VIDEO_MODEL_CACHE
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 model_dir", initial)
        if selected:
            self.lingbot_video_model_edit.setText(selected)

    def _on_lingbot_video_cam_clicked(self) -> None:
        self._refresh_lingbot_video_topic_combo()
        picked = self._pick_preview_color_source(
            getattr(self, "lingbot_video_topic_combo", None)
        )
        if picked is None:
            self._append_lingbot_video_log(
                "[ERROR] 无可用彩色图像：请在「预览图像」中选择或点击图像预览"
            )
            self._on_lingbot_video_status("无可用相机图像")
            return
        topic, image, _panel = picked
        out_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_VIDEO_CACHE_DIR, "camera_frames")
        _eai_platform_bootstrap.os.makedirs(out_dir, exist_ok=True)
        path = _eai_platform_bootstrap.os.path.join(out_dir, "ti2v_first_frame.png")
        bgr = _eai_platform_bootstrap.np.asarray(image)
        if bgr.ndim == 2:
            bgr = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
        _eai_platform_bootstrap.cv2.imwrite(path, bgr)
        self.lingbot_video_image_edit.setText(path)
        self.lingbot_video_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
        self._append_lingbot_video_log(f"已抓首帧 {topic} -> {path}")

    def _on_lingbot_video_run_clicked(self) -> None:
        if self._lingbot_video_launcher.is_running():
            return
        mode = str(self.lingbot_video_mode_combo.currentData() or "t2v")
        prompt_json = self.lingbot_video_prompt_edit.text().strip()
        if not prompt_json or not _eai_platform_bootstrap.os.path.isfile(prompt_json):
            self._append_lingbot_video_log(f"[ERROR] prompt.json 不存在: {prompt_json}")
            self._on_lingbot_video_status("prompt.json 不存在")
            return
        image = self.lingbot_video_image_edit.text().strip()
        if mode == "ti2v" and (not image or not _eai_platform_bootstrap.os.path.isfile(image)):
            # 自动从所选预览抓一帧
            self._on_lingbot_video_cam_clicked()
            image = self.lingbot_video_image_edit.text().strip()
        if mode == "ti2v" and (not image or not _eai_platform_bootstrap.os.path.isfile(image)):
            self._append_lingbot_video_log(
                "[ERROR] TI2V 需要有效首帧：请选择预览图像后点「抓取预览」"
            )
            self._on_lingbot_video_status("缺少首帧")
            return

        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.LINGBOT_VIDEO_OUTPUT_DIR, exist_ok=True)
        stamp = _eai_platform_bootstrap.time.strftime("%Y%m%d_%H%M%S")
        ext = ".png" if mode == "t2i" else ".mp4"
        output = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_VIDEO_OUTPUT_DIR, f"{mode}_{stamp}{ext}")
        self.lingbot_video_output_label.set_source_pixmap(None)
        if getattr(self, "lingbot_video_player", None) is not None:
            self.lingbot_video_player.close_video()
            self.lingbot_video_player.setVisible(False)
        self.lingbot_video_output_label.setVisible(True)
        self._lingbot_video_last_output = ""
        self.lingbot_video_open_btn.setEnabled(False)

        frames = int(self.lingbot_video_frames_spin.value())
        if mode == "t2i":
            frames = 1
        elif frames > 1 and (frames - 1) % 4 != 0:
            # snap down to valid 4n+1
            frames = max(1, ((frames - 1) // 4) * 4 + 1)
            self.lingbot_video_frames_spin.setValue(frames)
            self._append_lingbot_video_log(f"frames 已对齐为 4n+1: {frames}")

        self._lingbot_video_launcher.start(
            mode=mode,
            output=output,
            prompt_json=prompt_json,
            image=image if mode == "ti2v" else "",
            model_dir=self.lingbot_video_model_edit.text().strip(),
            height=int(self.lingbot_video_h_spin.value()),
            width=int(self.lingbot_video_w_spin.value()),
            num_frames=frames,
            steps=int(self.lingbot_video_steps_spin.value()),
            seed=int(self.lingbot_video_seed_spin.value()),
            run_refiner=bool(self.lingbot_video_refiner_check.isChecked()),
        )
        self._update_lingbot_video_ui()

    def _on_lingbot_video_stop_clicked(self) -> None:
        self._append_lingbot_video_log("--- 用户停止 LingBot-Video ---")
        self._lingbot_video_launcher.stop()
        self._update_lingbot_video_ui()

    def _lingbot_video_preview_output(self, path: str) -> None:
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            return
        lower = path.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            if getattr(self, "lingbot_video_player", None) is not None:
                self.lingbot_video_player.close_video()
                self.lingbot_video_player.setVisible(False)
            self.lingbot_video_output_label.setVisible(True)
            self.lingbot_video_output_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
            return
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            self.lingbot_video_output_label.setVisible(False)
            self.lingbot_video_output_label.set_source_pixmap(None)
            player = getattr(self, "lingbot_video_player", None)
            if player is None:
                self._append_lingbot_video_log("[WARN] 视频播放器未初始化")
                return
            player.setVisible(True)
            if player.load(path):
                player.play()
                self._append_lingbot_video_log(f"已加载视频并开始播放: {path}")
            else:
                self._append_lingbot_video_log(f"[WARN] 无法打开视频: {path}")

    def _on_lingbot_video_result(self, result: dict) -> None:
        if not result.get("ok"):
            return
        output = str(result.get("output") or "")
        self._lingbot_video_last_output = output
        self._lingbot_video_preview_output(output)
        self.lingbot_video_open_btn.setEnabled(bool(output and _eai_platform_bootstrap.os.path.isfile(output)))
        self._append_lingbot_video_log(
            f"完成 mode={result.get('mode')} "
            f"{result.get('width')}x{result.get('height')} "
            f"frames={result.get('num_frames')} steps={result.get('steps')} "
            f"-> {output}"
        )

    def _on_lingbot_video_open_clicked(self) -> None:
        path = getattr(self, "_lingbot_video_last_output", "")
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            return
        lower = path.lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            # 内嵌播放器再播一次；同时弹窗方便放大浏览
            if getattr(self, "lingbot_video_player", None) is not None:
                self.lingbot_video_output_label.setVisible(False)
                self.lingbot_video_player.setVisible(True)
                if self.lingbot_video_player.load(path):
                    self.lingbot_video_player.play()
            dlg = _eai_ui_widgets_video.SimEvalVideoDialog(path, self)
            dlg.exec_()
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(path))

    def _on_lingbot_world_status(self, msg: str) -> None:
        if hasattr(self, "lingbot_world_status_label"):
            self.lingbot_world_status_label.setText(msg)
        self.status_bar.showMessage(msg)

    def _on_lingbot_world_example_changed(self, *_args) -> None:
        ex = str(self.lingbot_world_example_combo.currentData() or "03")
        if ex != "custom":
            action = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_WORLD_ROOT_DEFAULT, "examples", ex)
            image = _eai_platform_bootstrap.os.path.join(action, "image.jpg")
            self.lingbot_world_action_edit.setText(action)
            self.lingbot_world_image_edit.setText(image)
            if _eai_platform_bootstrap.os.path.isfile(image):
                self.lingbot_world_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(image))
        self._update_lingbot_world_ui()

    def _on_lingbot_world_image_browse_clicked(self) -> None:
        current = self.lingbot_world_image_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isfile(current) else _eai_configuration_tabs.LINGBOT_WORLD_ROOT_DEFAULT
        selected, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择输入图像",
            initial,
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)",
        )
        if selected:
            self.lingbot_world_image_edit.setText(selected)
            self.lingbot_world_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(selected))

    def _on_lingbot_world_action_browse_clicked(self) -> None:
        current = self.lingbot_world_action_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isdir(current) else _eai_configuration_tabs.LINGBOT_WORLD_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 action 目录", initial)
        if selected:
            self.lingbot_world_action_edit.setText(selected)

    def _on_lingbot_world_ckpt_browse_clicked(self) -> None:
        current = self.lingbot_world_ckpt_edit.text().strip()
        initial = current if _eai_platform_bootstrap.os.path.isdir(current) else _eai_configuration_tabs.LINGBOT_WORLD_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择 ckpt 目录", initial)
        if selected:
            self.lingbot_world_ckpt_edit.setText(selected)

    def _on_lingbot_world_cam_clicked(self) -> None:
        self._refresh_lingbot_world_topic_combo()
        picked = self._pick_preview_color_source(
            getattr(self, "lingbot_world_topic_combo", None)
        )
        if picked is None:
            self._append_lingbot_world_log(
                "[ERROR] 无可用彩色图像：请在「预览图像」中选择或点击图像预览"
            )
            self._on_lingbot_world_status("无可用相机图像")
            return
        topic, image, _panel = picked
        out_dir = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_WORLD_CACHE_DIR, "camera_input")
        _eai_platform_bootstrap.os.makedirs(out_dir, exist_ok=True)
        path = _eai_platform_bootstrap.os.path.join(out_dir, "image.jpg")
        bgr = _eai_platform_bootstrap.np.asarray(image)
        if bgr.ndim == 2:
            bgr = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
        _eai_platform_bootstrap.cv2.imwrite(path, bgr, [int(_eai_platform_bootstrap.cv2.IMWRITE_JPEG_QUALITY), 95])
        # keep current action_path (example poses); switch combo to custom
        idx = self.lingbot_world_example_combo.findData("custom")
        if idx >= 0:
            self.lingbot_world_example_combo.setCurrentIndex(idx)
        self.lingbot_world_image_edit.setText(path)
        self.lingbot_world_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(path))
        if not self.lingbot_world_action_edit.text().strip():
            self.lingbot_world_action_edit.setText(
                _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_WORLD_ROOT_DEFAULT, "examples", "03")
            )
        self._append_lingbot_world_log(f"已抓帧 {topic} -> {path}")
        self._update_lingbot_world_ui()

    def _on_lingbot_world_run_clicked(self) -> None:
        if self._lingbot_world_launcher.is_running():
            return
        image = self.lingbot_world_image_edit.text().strip()
        action = self.lingbot_world_action_edit.text().strip()
        if not image or not _eai_platform_bootstrap.os.path.isfile(image):
            self._append_lingbot_world_log(f"[ERROR] 图像不存在: {image}")
            return
        if not action or not _eai_platform_bootstrap.os.path.isdir(action):
            self._append_lingbot_world_log(f"[ERROR] action 目录不存在: {action}")
            return
        frames = int(self.lingbot_world_frames_spin.value())
        if frames > 1 and (frames - 1) % 4 != 0:
            frames = max(5, ((frames - 1) // 4) * 4 + 1)
            self.lingbot_world_frames_spin.setValue(frames)
            self._append_lingbot_world_log(f"frames 已对齐为 4n+1: {frames}")

        _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.LINGBOT_WORLD_OUTPUT_DIR, exist_ok=True)
        stamp = _eai_platform_bootstrap.time.strftime("%Y%m%d_%H%M%S")
        output = _eai_platform_bootstrap.os.path.join(_eai_configuration_tabs.LINGBOT_WORLD_OUTPUT_DIR, f"world_{stamp}.mp4")
        if getattr(self, "lingbot_world_player", None) is not None:
            self.lingbot_world_player.close_video()
        self._lingbot_world_last_output = ""
        self.lingbot_world_play_btn.setEnabled(False)
        self.lingbot_world_input_label.set_source_pixmap(_eai_platform_bootstrap.QPixmap(image))

        self._lingbot_world_launcher.start(
            image=image,
            action_path=action,
            prompt=self.lingbot_world_prompt_edit.toPlainText().strip()
            or _eai_configuration_tabs.LINGBOT_WORLD_PROMPT_DEFAULT,
            output=output,
            ckpt_dir=self.lingbot_world_ckpt_edit.text().strip(),
            size=str(self.lingbot_world_size_combo.currentData() or "480*832"),
            frame_num=frames,
            seed=int(self.lingbot_world_seed_spin.value()),
            t5_cpu=bool(self.lingbot_world_t5_cpu_check.isChecked()),
            offload_model=bool(self.lingbot_world_offload_check.isChecked()),
        )
        self._update_lingbot_world_ui()

    def _on_lingbot_world_stop_clicked(self) -> None:
        self._append_lingbot_world_log("--- 用户停止 LingBot-World ---")
        self._lingbot_world_launcher.stop()
        self._update_lingbot_world_ui()

    def _on_lingbot_world_result(self, result: dict) -> None:
        if not result.get("ok"):
            return
        output = str(result.get("output") or "")
        self._lingbot_world_last_output = output
        self.lingbot_world_play_btn.setEnabled(bool(output and _eai_platform_bootstrap.os.path.isfile(output)))
        if output and _eai_platform_bootstrap.os.path.isfile(output) and getattr(self, "lingbot_world_player", None):
            if self.lingbot_world_player.load(output):
                self.lingbot_world_player.play()
        self._append_lingbot_world_log(
            f"完成 frames={result.get('frame_num')} size={result.get('size')} "
            f"-> {output}"
        )

    def _on_lingbot_world_play_clicked(self) -> None:
        path = getattr(self, "_lingbot_world_last_output", "")
        if not path or not _eai_platform_bootstrap.os.path.isfile(path):
            return
        if getattr(self, "lingbot_world_player", None) is not None:
            if self.lingbot_world_player.load(path):
                self.lingbot_world_player.play()
        dlg = _eai_ui_widgets_video.SimEvalVideoDialog(path, self)
        dlg.exec_()

