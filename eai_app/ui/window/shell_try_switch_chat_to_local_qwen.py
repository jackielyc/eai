"""ui.window.shell try switch chat to local qwen — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_adapters_models_qwen_remote = module_ref('eai_app.adapters.models.qwen_remote')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_segmentation = module_ref('eai_app.perception.segmentation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')
_eai_ui_datasets_notes = module_ref('eai_app.ui.datasets.notes')
_eai_ui_widgets_layouts = module_ref('eai_app.ui.widgets.layouts')

class ShellTrySwitchChatToLocalQwenMixin:
    def _try_switch_chat_to_local_qwen(self) -> None:
        if self._selected_test_qwen_target() != "local":
            return
        api = self._local_qwen_launcher.api_base()
        info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api)
        if info and info.get("ok"):
            mid = str(info.get("model") or self._local_qwen_launcher.last_model_id())
            self.chat_panel.apply_local_qwen_service_preset(
                api_base=api, model_id=mid, silent=False
            )
            self._update_test_qwen_ui()
            return

        if _eai_adapters_models_qwen_local.check_local_qwen_hostctl_health():
            _ok, body = _eai_adapters_models_qwen_local.local_qwen_hostctl_request("/status", method="GET", timeout_s=2.0)
            if _ok and not body.get("process_running") and not body.get("service_healthy"):
                self._local_qwen_launcher._starting = False
                if not self._local_qwen_launcher._fail_notified:
                    self._local_qwen_launcher._fail_notified = True
                    err = str(body.get("last_error") or "推理进程已退出")
                    self._append_test_infer_log(f"启动失败: {err}")
                self._update_test_qwen_ui()
                return

        self._qwen_switch_tries = getattr(self, "_qwen_switch_tries", 0) + 1
        if self._qwen_switch_tries < 180:
            _eai_platform_bootstrap.QTimer.singleShot(2000, self._try_switch_chat_to_local_qwen)
        else:
            self._local_qwen_launcher._starting = False
            self._append_test_infer_log(
                "启动超时：模型仍未就绪，请查看 eai/log/local_qwen_service.log"
            )
        self._update_test_qwen_ui()

    def _try_switch_chat_to_remote_qwen(self) -> None:
        if self._selected_test_qwen_target() != "remote":
            return
        api = self._remote_qwen_launcher.api_base()
        info = _eai_adapters_models_qwen_local.fetch_local_qwen_server_info(api)
        if info and info.get("ok"):
            self._remote_qwen_launcher._starting = False
            mid = str(info.get("model") or self._remote_qwen_launcher.last_model_id())
            self.chat_panel.apply_remote_qwen_service_preset(
                api_base=api, model_id=mid, silent=False
            )
            self._update_test_qwen_ui()
            return

        if (
            not getattr(self._remote_qwen_launcher, "_starting", False)
            and getattr(self._remote_qwen_launcher, "_fail_notified", False)
        ):
            self._update_test_qwen_ui()
            return

        self._qwen_switch_tries = getattr(self, "_qwen_switch_tries", 0) + 1
        # 远程 35B 多卡加载可能更久
        if self._qwen_switch_tries < 300:
            _eai_platform_bootstrap.QTimer.singleShot(2000, self._try_switch_chat_to_remote_qwen)
        else:
            self._remote_qwen_launcher._starting = False
            host_id = self._selected_test_qwen_remote_host()
            prof = _eai_adapters_models_qwen_remote.remote_qwen_profile(host_id)
            log_hint = str(prof.get("remote_work") or "") + "/local_qwen_service.log"
            self._append_test_infer_log(
                f"远程启动超时（{host_id}）：请查看远程日志 {log_hint}"
            )
        self._update_test_qwen_ui()

    def _on_test_qwen_stop_clicked(self) -> None:
        if self._selected_test_qwen_target() == "remote":
            self._remote_qwen_launcher.stop(
                host_id=self._selected_test_qwen_remote_host()
            )
        else:
            self._local_qwen_launcher.stop()
        self._update_test_qwen_ui()

    def _on_cad_mode_changed(self, *_args) -> None:
        mode = str(self.cad_mode_combo.currentData() or "photos")
        if mode == "mesh":
            self.cad_src_label.setText("Mesh 文件:")
            self.cad_src_edit.setPlaceholderText("选择 .obj / .ply / .glb / .stl")
            self.cad_poisson_spin.setEnabled(False)
            self.cad_capture_btn.setEnabled(False)
            self.cad_capture_reset_btn.setEnabled(False)
        else:
            self.cad_src_label.setText("照片目录:")
            self.cad_src_edit.setPlaceholderText("选择多视角照片目录，或用下方「拍一张」采集")
            self.cad_poisson_spin.setEnabled(True)
            self._update_cad_ui()

    def _cad_photos_dir_for_name(self, name: _eai_platform_bootstrap.Optional[str] = None) -> str:
        obj = (name or self.cad_name_edit.text() or "my_object").strip().replace(" ", "_")
        for ch in '/\\:*?"<>|':
            obj = obj.replace(ch, "_")
        if not obj:
            obj = "my_object"
        return _eai_platform_bootstrap.os.path.join(_eai_configuration_policies.CAD_MESHES_DIR, obj, "photos")

    def _count_photos_in_dir(self, photos_dir: str) -> int:
        if not _eai_platform_bootstrap.os.path.isdir(photos_dir):
            return 0
        n = 0
        for name in _eai_platform_bootstrap.os.listdir(photos_dir):
            lower = name.lower()
            if "_depth" in lower:
                continue
            if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                n += 1
        return n

    def _sync_cad_capture_from_disk(self) -> None:
        photos_dir = self._cad_photos_dir_for_name()
        self._cad_capture_dir = photos_dir
        self._cad_capture_saved = self._count_photos_in_dir(photos_dir)
        if self._cad_capture_saved > 0 and str(
            self.cad_mode_combo.currentData() or "photos"
        ) == "photos":
            self.cad_src_edit.setText(photos_dir)
        self._update_cad_capture_btn()

    def _update_cad_capture_btn(self, *_args) -> None:
        target = int(self.cad_capture_target_spin.value())
        saved = int(self._cad_capture_saved)
        self.cad_capture_btn.setText(f"拍一张 ({saved}/{target})")
        if saved >= target:
            self.cad_capture_hint.setText(f"已满 {target} 张，可点「生成 CAD」或勾选自动生成")
            self.cad_capture_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        elif saved > 0:
            self.cad_capture_hint.setText(
                f"已拍 {saved} 张 — 请换视角后再拍（相邻约 60% 重叠）"
            )
            self.cad_capture_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        else:
            self.cad_capture_hint.setText(
                "头部 RGB-D：每拍一张请转动物体，视角差太小会被拒绝"
            )
            self.cad_capture_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")

    def _cad_frame_too_similar(self, bgr: _eai_platform_bootstrap.np.ndarray) -> _eai_platform_bootstrap.Tuple[bool, float, float]:
        """与上一张比较，返回 (太相似?, corr, mae)。"""
        gray = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        gray = _eai_platform_bootstrap.cv2.resize(gray, (320, 200), interpolation=_eai_platform_bootstrap.cv2.INTER_AREA)
        if self._cad_last_capture_gray is None:
            return False, 0.0, 999.0
        a = self._cad_last_capture_gray.astype(_eai_platform_bootstrap.np.float32)
        b = gray.astype(_eai_platform_bootstrap.np.float32)
        mae = float(_eai_platform_bootstrap.np.mean(_eai_platform_bootstrap.np.abs(a - b)))
        aa = (a - a.mean()).ravel()
        bb = (b - b.mean()).ravel()
        corr = float((aa @ bb) / (_eai_platform_bootstrap.np.linalg.norm(aa) * _eai_platform_bootstrap.np.linalg.norm(bb) + 1e-9))
        too_similar = corr >= _eai_configuration_policies.CAD_CAPTURE_MIN_DIFF_CORR or (
            corr >= 0.96 and mae < _eai_configuration_policies.CAD_CAPTURE_MIN_DIFF_MAE
        )
        return too_similar, corr, mae

    def _ensure_head_rgbd_subscribed(self) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.Optional[str], _eai_platform_bootstrap.Optional[str]]:
        """确保头部彩色+深度已勾选，返回 (color_topic, depth_topic)。"""
        known = list(self._topic_types.keys()) or list(self.topic_checks.keys())
        color = _eai_perception_image_geometry.resolve_head_color_topic(known) or _eai_configuration_policies.CAD_CAPTURE_TOPIC_DEFAULT
        depth = _eai_perception_image_geometry.find_paired_depth_topic(color, known) or _eai_configuration_policies.CAD_CAPTURE_DEPTH_DEFAULT

        changed = False
        for topic in (color, depth):
            checkbox = self.topic_checks.get(topic)
            if checkbox is not None and not checkbox.isChecked():
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                changed = True
                self._append_cad_log(f"已自动勾选: {topic}")
        if changed:
            enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
            self._apply_selection(enabled)
        return color, depth

    def _pick_head_color_source(
        self,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Optional[_eai_ui_cameras_panels.CameraPanel]]]:
        """CAD 采集专用：只取头部彩色相机画面。"""
        color_topic, _depth_topic = self._ensure_head_rgbd_subscribed()
        if color_topic is None:
            return None

        panel = self.panels.get(color_topic)
        if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and panel._latest_image is not None:
            return color_topic, panel._latest_image.copy(), panel

        image = self._frame_cache.get(color_topic)
        if image is not None and _eai_platform_bootstrap.np.asarray(image).ndim >= 2:
            return (
                color_topic,
                _eai_platform_bootstrap.np.asarray(image).copy(),
                panel if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) else None,
            )

        for name, cached in self._frame_cache.items():
            if not _eai_perception_image_geometry.is_head_color_topic(name):
                continue
            if cached is None or _eai_platform_bootstrap.np.asarray(cached).ndim < 2:
                continue
            p = self.panels.get(name)
            return (
                name,
                _eai_platform_bootstrap.np.asarray(cached).copy(),
                p if isinstance(p, _eai_ui_cameras_panels.CameraPanel) else None,
            )
        return None

    def _pick_head_depth_frame(self, color_topic: str) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
        known = list(self._frame_cache.keys()) + list(self.topic_checks.keys())
        depth_topic = _eai_perception_image_geometry.find_paired_depth_topic(color_topic, known) or _eai_configuration_policies.CAD_CAPTURE_DEPTH_DEFAULT
        depth = self._frame_cache.get(depth_topic)
        if depth is None:
            panel = self.panels.get(depth_topic)
            if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D) and getattr(panel, "_latest_depth", None) is not None:
                depth = panel._latest_depth
        if depth is None:
            return None
        arr = _eai_platform_bootstrap.np.asarray(depth)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

    def _save_cad_intrinsics(self, photos_dir: str, color_topic: str, width: int, height: int) -> None:
        depth_topic = _eai_perception_image_geometry.find_paired_depth_topic(
            color_topic, list(self._frame_cache.keys()) + list(self.topic_checks.keys())
        ) or _eai_configuration_policies.CAD_CAPTURE_DEPTH_DEFAULT
        fx, fy, cx, cy = self.node.get_intrinsics(depth_topic, width, height)
        # 若只有 color 的 CameraInfo，再试 color
        if abs(fx - 0.9 * max(width, height)) < 1e-3:
            fx, fy, cx, cy = self.node.get_intrinsics(color_topic, width, height)
        meta = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(width),
            "height": int(height),
            "depth_scale": 1000.0,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
        }
        path = _eai_platform_bootstrap.os.path.join(photos_dir, "intrinsics.json")
        with open(path, "w", encoding="utf-8") as f:
            _eai_platform_bootstrap.json.dump(meta, f, indent=2)

    def _ensure_cad_capture_session(self) -> str:
        photos_dir = self._cad_photos_dir_for_name()
        if self._cad_capture_dir != photos_dir:
            self._cad_capture_dir = photos_dir
            self._cad_capture_saved = self._count_photos_in_dir(photos_dir)
            self._cad_last_capture_gray = None
        _eai_platform_bootstrap.os.makedirs(photos_dir, exist_ok=True)
        return photos_dir

    def _on_cad_capture_clicked(self) -> None:
        if self._cad_launcher.is_running():
            self.status_bar.showMessage("CAD 重建进行中，请稍候再拍")
            return
        if str(self.cad_mode_combo.currentData() or "photos") != "photos":
            self.cad_mode_combo.setCurrentIndex(self.cad_mode_combo.findData("photos"))

        source = self._pick_head_color_source()
        if source is None:
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "CAD 采集",
                "未收到头部相机画面。\n\n"
                f"请确认已发布 {_eai_configuration_policies.CAD_CAPTURE_TOPIC_DEFAULT}，\n"
                "并在左侧勾选该 topic 后等待图像到达。",
            )
            return

        topic, image, _panel = source
        bgr = _eai_platform_bootstrap.np.asarray(image)
        if bgr.ndim == 2:
            bgr = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
        elif bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_BGRA2BGR)

        too_similar, corr, mae = self._cad_frame_too_similar(bgr)
        if too_similar:
            msg = (
                f"视角变化太小（corr={corr:.3f}, mae={mae:.1f}），未保存。\n"
                "请转动物体（或移动头部）后再拍一张。"
            )
            self._append_cad_log(msg.replace("\n", " "))
            self.status_bar.showMessage(msg.split("\n")[0])
            self.cad_capture_hint.setText(msg.split("\n")[0])
            self.cad_capture_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
            return

        depth = self._pick_head_depth_frame(topic)
        if depth is None:
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "CAD 采集",
                "未收到头部深度图。\n\n"
                f"请确认已发布并勾选 {_eai_configuration_policies.CAD_CAPTURE_DEPTH_DEFAULT}。\n"
                "RGB-D 重建需要彩色+深度。",
            )
            return

        photos_dir = self._ensure_cad_capture_session()
        idx = self._cad_capture_saved + 1
        color_path = _eai_platform_bootstrap.os.path.join(photos_dir, f"img_{idx:04d}.jpg")
        depth_path = _eai_platform_bootstrap.os.path.join(photos_dir, f"img_{idx:04d}_depth.png")
        while _eai_platform_bootstrap.os.path.isfile(color_path):
            idx += 1
            color_path = _eai_platform_bootstrap.os.path.join(photos_dir, f"img_{idx:04d}.jpg")
            depth_path = _eai_platform_bootstrap.os.path.join(photos_dir, f"img_{idx:04d}_depth.png")

        # 深度对齐到彩色分辨率
        depth_arr = _eai_platform_bootstrap.np.asarray(depth)
        if depth_arr.shape[:2] != bgr.shape[:2]:
            depth_arr = _eai_platform_bootstrap.cv2.resize(
                depth_arr, (bgr.shape[1], bgr.shape[0]), interpolation=_eai_platform_bootstrap.cv2.INTER_NEAREST
            )
        if depth_arr.dtype == _eai_platform_bootstrap.np.float32 or depth_arr.dtype == _eai_platform_bootstrap.np.float64:
            # 米 → mm
            depth_u16 = _eai_platform_bootstrap.np.clip(depth_arr * 1000.0, 0, 65535).astype(_eai_platform_bootstrap.np.uint16)
        else:
            depth_u16 = depth_arr.astype(_eai_platform_bootstrap.np.uint16)

        ok_c = _eai_platform_bootstrap.cv2.imwrite(color_path, bgr, [int(_eai_platform_bootstrap.cv2.IMWRITE_JPEG_QUALITY), 95])
        ok_d = _eai_platform_bootstrap.cv2.imwrite(depth_path, depth_u16)
        if not ok_c or not ok_d:
            _eai_platform_bootstrap.QMessageBox.warning(self, "CAD 采集", f"保存失败: {color_path}")
            return

        self._save_cad_intrinsics(photos_dir, topic, bgr.shape[1], bgr.shape[0])
        gray = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_BGR2GRAY)
        self._cad_last_capture_gray = _eai_platform_bootstrap.cv2.resize(gray, (320, 200), interpolation=_eai_platform_bootstrap.cv2.INTER_AREA)
        self._cad_capture_saved = self._count_photos_in_dir(photos_dir)
        self.cad_src_edit.setText(photos_dir)
        self._update_cad_capture_btn()
        self._append_cad_log(
            f"头部 RGB-D [{self._cad_capture_saved}/{int(self.cad_capture_target_spin.value())}] "
            f"{_eai_platform_bootstrap.os.path.basename(color_path)} + depth  ← {topic}  "
            f"({bgr.shape[1]}x{bgr.shape[0]}, Δcorr={corr:.3f})"
        )
        self.status_bar.showMessage(
            f"头部 RGB-D 已拍 {self._cad_capture_saved}/{int(self.cad_capture_target_spin.value())} 张"
        )

        target = int(self.cad_capture_target_spin.value())
        if self._cad_capture_saved >= target and self.cad_auto_gen_check.isChecked():
            self._append_cad_log(f"已满 {target} 张，开始自动生成 CAD（RGB-D TSDF）…")
            self._on_cad_start_clicked()

    def _on_cad_capture_reset_clicked(self) -> None:
        photos_dir = self._cad_photos_dir_for_name()
        if _eai_platform_bootstrap.os.path.isdir(photos_dir):
            removed = 0
            for name in _eai_platform_bootstrap.os.listdir(photos_dir):
                lower = name.lower()
                if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".json")):
                    try:
                        _eai_platform_bootstrap.os.remove(_eai_platform_bootstrap.os.path.join(photos_dir, name))
                        removed += 1
                    except OSError:
                        pass
            self._append_cad_log(f"已清空采集目录 ({removed} 文件): {photos_dir}")
        self._cad_capture_dir = photos_dir
        self._cad_capture_saved = 0
        self._cad_last_capture_gray = None
        self._update_cad_capture_btn()
        self.status_bar.showMessage("采集已清空")

    def _on_cad_src_browse_clicked(self) -> None:
        mode = str(self.cad_mode_combo.currentData() or "photos")
        current = self.cad_src_edit.text().strip()
        if mode == "mesh":
            initial = current if _eai_platform_bootstrap.os.path.isfile(current) else (
                _eai_platform_bootstrap.os.path.dirname(current) if current else _eai_configuration_policies.CAD_MESHES_DIR
            )
            if not _eai_platform_bootstrap.os.path.isdir(initial):
                initial = _eai_platform_bootstrap.os.path.expanduser("~")
            path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
                self,
                "选择 mesh 文件",
                initial,
                "Mesh (*.obj *.ply *.glb *.stl *.off);;All Files (*)",
            )
            if path:
                self.cad_src_edit.setText(path)
                base = _eai_platform_bootstrap.os.path.splitext(_eai_platform_bootstrap.os.path.basename(path))[0]
                if not self.cad_name_edit.text().strip() or self.cad_name_edit.text() in (
                    "my_object",
                    "imported_object",
                ):
                    self.cad_name_edit.setText(base or "imported_object")
        else:
            initial = current if _eai_platform_bootstrap.os.path.isdir(current) else _eai_platform_bootstrap.os.path.expanduser("~")
            path = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(self, "选择多视角照片目录", initial)
            if path:
                self.cad_src_edit.setText(path)
                base = _eai_platform_bootstrap.os.path.basename(path.rstrip(_eai_platform_bootstrap.os.sep))
                if not self.cad_name_edit.text().strip() or self.cad_name_edit.text() == "my_object":
                    self.cad_name_edit.setText(base or "my_object")
                self._cad_capture_dir = path
                self._cad_capture_saved = self._count_photos_in_dir(path)
                self._update_cad_capture_btn()

    def _refresh_cad_mesh_combo(self) -> None:
        current = self.cad_existing_combo.currentData()
        self.cad_existing_combo.blockSignals(True)
        self.cad_existing_combo.clear()
        meshes = _eai_configuration_policies.list_reconstructed_meshes()
        if not meshes:
            self.cad_existing_combo.addItem("(暂无 reconstructed.obj)", "")
        else:
            for name, path in meshes:
                self.cad_existing_combo.addItem(f"{name}", path)
        if current:
            idx = self.cad_existing_combo.findData(current)
            if idx >= 0:
                self.cad_existing_combo.setCurrentIndex(idx)
        self.cad_existing_combo.blockSignals(False)

    def _append_cad_log(self, line: str) -> None:
        self.cad_log_edit.append(line)
        scrollbar = self.cad_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_cad_ui(self, *_args) -> None:
        running = self._cad_launcher.is_running()
        mode_photos = str(self.cad_mode_combo.currentData() or "photos") == "photos"
        self.cad_stop_btn.setEnabled(running)
        self.cad_start_btn.setEnabled(not running)
        self.cad_capture_btn.setEnabled(not running and mode_photos)
        self.cad_capture_reset_btn.setEnabled(not running and mode_photos)
        if running:
            self.cad_status_label.setText("CAD: 运行中")
            self.cad_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.cad_status_label.setText("CAD: 空闲")
            self.cad_status_label.setStyleSheet("")
        self._update_cad_capture_btn()

    def _on_cad_start_clicked(self) -> None:
        name = self.cad_name_edit.text().strip() or "my_object"
        src = self.cad_src_edit.text().strip()
        extent = float(self.cad_extent_spin.value())
        mode = str(self.cad_mode_combo.currentData() or "photos")
        if not src:
            # 若已采集照片，自动用 photos 目录
            photos_dir = self._cad_photos_dir_for_name(name)
            if self._count_photos_in_dir(photos_dir) > 0:
                src = photos_dir
                self.cad_src_edit.setText(src)
        if not src:
            _eai_platform_bootstrap.QMessageBox.information(
                self,
                "CAD",
                "请先用「拍一张」采集照片，或选择照片目录 / mesh 文件。",
            )
            return
        if mode == "mesh":
            self._cad_launcher.start_from_mesh(src, name, target_extent_m=extent)
        else:
            n_photos = self._count_photos_in_dir(src) if _eai_platform_bootstrap.os.path.isdir(src) else 0
            if 0 < n_photos < _eai_configuration_policies.CAD_MIN_IMAGES_DEFAULT:
                _eai_platform_bootstrap.QMessageBox.information(
                    self,
                    "CAD",
                    f"当前仅 {n_photos} 张照片，至少需要 {_eai_configuration_policies.CAD_MIN_IMAGES_DEFAULT} 张。\n"
                    f"请继续点「拍一张」换视角采集。",
                )
                return
            self._cad_launcher.start_from_images(
                src,
                name,
                target_extent_m=extent,
                poisson_depth=int(self.cad_poisson_spin.value()),
                min_images=_eai_configuration_policies.CAD_MIN_IMAGES_DEFAULT,
            )
        self._update_cad_ui()

    def _on_cad_stop_clicked(self) -> None:
        self._cad_launcher.stop()

    def _on_cad_clear_log_clicked(self) -> None:
        self.cad_log_edit.clear()

    def _on_cad_apply_fp_clicked(self) -> None:
        mesh = str(self.cad_existing_combo.currentData() or "").strip()
        if not mesh:
            mesh = self._cad_launcher.last_mesh()
        if not mesh or not _eai_platform_bootstrap.os.path.isfile(mesh):
            _eai_platform_bootstrap.QMessageBox.information(self, "CAD", "没有可用的 reconstructed.obj。")
            return
        self.fp_mesh_edit.setText(mesh)
        idx = self.pose_backend_combo.findData(_eai_configuration_perception.POSE_BACKEND_FOUNDATIONPOSE)
        if idx >= 0:
            self.pose_backend_combo.setCurrentIndex(idx)
        self._on_pose_settings_changed()
        self.status_bar.showMessage(f"已应用到 FP mesh: {mesh}")

    def _on_cad_mesh_ready(self, mesh_path: str) -> None:
        self._refresh_cad_mesh_combo()
        idx = self.cad_existing_combo.findData(mesh_path)
        if idx < 0:
            # findData 可能因路径规范化不一致失败
            abs_mesh = _eai_platform_bootstrap.os.path.abspath(mesh_path)
            for i in range(self.cad_existing_combo.count()):
                if _eai_platform_bootstrap.os.path.abspath(str(self.cad_existing_combo.itemData(i) or "")) == abs_mesh:
                    idx = i
                    break
        if idx >= 0:
            self.cad_existing_combo.setCurrentIndex(idx)
        self.fp_mesh_edit.setText(mesh_path)
        self._on_pose_settings_changed()
        self._append_cad_log(f"已自动填入分割页 FP mesh: {mesh_path}")

    def _update_local_ai_deploy_btn(self, *_args) -> None:
        running = self._local_ai_launcher.is_sam3_running()
        if running:
            self.local_ai_deploy_btn.setText("停止 AI")
            self.local_ai_deploy_btn.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
        else:
            self.local_ai_deploy_btn.setText("本地部署 AI")
            self.local_ai_deploy_btn.setStyleSheet("")

    def _configure_sam3_for_local_http(self) -> None:
        url = _eai_perception_segmentation.resolve_sam3_viewer_server_url()
        idx = self.segment_backend_combo.findData(_eai_configuration_perception.SAM3_BACKEND_POINT)
        if idx >= 0:
            self.segment_backend_combo.setCurrentIndex(idx)
        self.sam3_http_check.setChecked(True)
        _eai_perception_segmentation.set_segment_settings(
            backend=_eai_configuration_perception.SAM3_BACKEND_POINT,
            sam3_text=self.sam3_text_edit.text().strip(),
            sam3_use_http=True,
            sam3_server_url=url,
        )
        self._on_segment_settings_changed()

    def _after_local_ai_deploy(self) -> None:
        self._update_local_ai_deploy_btn()
        self._refresh_sam3_status()
        url = _eai_perception_segmentation.resolve_sam3_viewer_server_url()
        if _eai_perception_segmentation.check_sam3_server_health(url):
            self.status_bar.showMessage(f"本地 SAM3 已在线: {url}")
        elif self._local_ai_launcher.is_sam3_running():
            self.status_bar.showMessage(f"SAM3 启动中，请稍候… ({url})")

    def closeEvent(self, event: _eai_platform_bootstrap.QCloseEvent) -> None:
        grasp_panel = getattr(self, "grasp_execution_panel", None)
        if grasp_panel is not None and (grasp_panel.reserved or grasp_panel.process is not None):
            grasp_panel.stop()
            self.status_bar.showMessage("正在停止抓取，请待停止反馈确认后再关闭窗口")
            event.ignore()
            return
        app = _eai_platform_bootstrap.QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        try:
            self._save_viewer_ui_state()
        except Exception as exc:
            print(f"[viewer_ui_state] save failed: {exc}", flush=True)
        self._stop_skeleton_tracking()
        self._stop_real_eval_local_cam()
        self._stop_local_cam_preview()
        if getattr(self, "_local_cam_mjpeg_launcher", None) is not None:
            self._local_cam_mjpeg_launcher.shutdown()
        if getattr(self, "_local_cam_tunnel_launcher", None) is not None:
            self._local_cam_tunnel_launcher.shutdown()
        self._stop_human_demo_playback()
        self._ui_timer.stop()
        self._sam3_health_timer.stop()
        self._fp_health_timer.stop()
        if hasattr(self, "_sim_frame_timer"):
            self._sim_frame_timer.stop()
        self._stack_launcher.shutdown()
        self._replay_launcher.shutdown()
        if getattr(self, "ctx_video_player", None) is not None:
            self.ctx_video_player.close_video()
        self._stop_ctx_rerun_embed()
        self._local_ai_launcher.shutdown()
        self._train_launcher.shutdown()
        self._cad_launcher.shutdown()
        self._sim_bridge_launcher.shutdown()
        self._sim_eval_launcher.shutdown()
        if getattr(self, "_lingbot_launcher", None) is not None:
            self._lingbot_launcher.shutdown()
        cancel = getattr(self, "_mv6d_lingbot_cancel", None)
        if cancel is not None:
            cancel.set()
        if getattr(self, "_lingbot_depth_launcher", None) is not None:
            self._lingbot_depth_launcher.shutdown()
        if getattr(self, "_reward_launcher", None) is not None:
            self._reward_launcher.shutdown()
        if getattr(self, "_reward_wf_launcher", None) is not None:
            self._reward_wf_launcher.shutdown()
        if getattr(self, "_dojo_rl_launcher", None) is not None:
            self._dojo_rl_launcher.shutdown()
        if getattr(self, "_robometer_launcher", None) is not None:
            self._robometer_launcher.shutdown()
        if getattr(self, "_rynnvalue_launcher", None) is not None:
            self._rynnvalue_launcher.shutdown()
        if getattr(self, "_roboicl_launcher", None) is not None:
            self._roboicl_launcher.shutdown()
        if getattr(self, "_mujoco_launcher", None) is not None:
            self._mujoco_launcher.shutdown()
        if getattr(self, "_lingbot_map_launcher", None) is not None:
            self._lingbot_map_launcher.shutdown()
        if getattr(self, "_lingbot_video_launcher", None) is not None:
            self._lingbot_video_launcher.shutdown()
        if getattr(self, "lingbot_video_player", None) is not None:
            self.lingbot_video_player.close_video()
        if getattr(self, "_lingbot_world_launcher", None) is not None:
            self._lingbot_world_launcher.shutdown()
        if getattr(self, "_bagel_launcher", None) is not None:
            if getattr(self, "_bagel_ready_poll_timer", None) is not None:
                self._bagel_ready_poll_timer.stop()
            self._bagel_launcher.shutdown()
        if getattr(self, "_bagel_api_launcher", None) is not None:
            if getattr(self, "_bagel_api_ready_poll_timer", None) is not None:
                self._bagel_api_ready_poll_timer.stop()
            self._bagel_api_launcher.shutdown()
        if getattr(self, "_humanego_launcher", None) is not None:
            self._humanego_launcher.shutdown()
        if getattr(self, "lingbot_world_player", None) is not None:
            self.lingbot_world_player.close_video()
        if getattr(self, "_real_eval_timer", None) is not None:
            self._real_eval_timer.stop()
        self._real_eval_active = False
        self._local_qwen_launcher.shutdown()
        self._remote_qwen_launcher.shutdown()
        for launcher_name in (
            "_robodojo_policy_launcher",
            "_pi_policy_launcher",
            "_psi_remote_launcher",
        ):
            launcher = getattr(self, launcher_name, None)
            if launcher is not None:
                launcher.shutdown()
        for panel in self.panels.values():
            if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D):
                panel.stop_robot_timer()
        try:
            self._stop_robot_tcp_bridge()
        except Exception:
            pass
        self.node.prepare_shutdown()
        super().closeEvent(event)

    def _start_rrd_replay(self, path: str, loop_count: int) -> bool:
        """启动 RRD 回放（回放页 / 上下文学习共用）。成功返回 True。"""
        robot, stack = self._stack_launcher.get_cached_status()
        if stack != "运行中":
            if self.node.is_hal_arm_ready() or robot in ("就绪", "启动中"):
                self._stack_launcher.start_stack()
                self.status_bar.showMessage(
                    "手/臂服务栈启动中，回放将自动切 model 模式并使能手/臂…",
                    8000,
                )
            else:
                reply = _eai_platform_bootstrap.QMessageBox.warning(
                    self,
                    "回放前置条件",
                    "手/臂服务栈未运行，且 HAL 未就绪。\n\n"
                    "请先点击「启动机器人栈」，或仍要继续仅启动回放节点？",
                    _eai_platform_bootstrap.QMessageBox.Yes | _eai_platform_bootstrap.QMessageBox.No,
                    _eai_platform_bootstrap.QMessageBox.No,
                )
                if reply != _eai_platform_bootstrap.QMessageBox.Yes:
                    return False

        warnings = self.node.prepare_for_rrd_replay()
        if warnings:
            reply = _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "回放前置条件",
                "当前未完全满足回放条件（将自动重试使能/切模式）：\n\n"
                + "\n".join(f"• {w}" for w in warnings)
                + "\n\n仍要继续启动回放节点？",
                _eai_platform_bootstrap.QMessageBox.Yes | _eai_platform_bootstrap.QMessageBox.No,
                _eai_platform_bootstrap.QMessageBox.Yes,
            )
            if reply != _eai_platform_bootstrap.QMessageBox.Yes:
                self.node._stop_replay_prep_mode_timer()
                return False

        self._replay_launcher.start(path, loop_count=loop_count)
        self._update_replay_ui()
        self._update_ctx_ui()
        return self._replay_launcher.is_running()

    def _load_ctx_default_video(self) -> None:
        path = _eai_configuration_policies.CTX_DEFAULT_VIDEO_PATH
        if not _eai_platform_bootstrap.os.path.isfile(path):
            return
        self._ctx_video_path = path
        self.ctx_video_path_edit.setText(path)
        self.ctx_video_path_edit.setToolTip(path)
        self.ctx_video_player.load(path)
        sibling = self._ctx_find_sibling(path, (".rrd",))
        if sibling:
            self._ctx_rrd_path = sibling
            self.ctx_rrd_path_edit.setText(sibling)
            self.ctx_rrd_path_edit.setToolTip(sibling)

    def _open_rrd_in_native_rerun(self, path: str) -> bool:
        argv = _eai_ui_datasets_notes.resolve_rerun_argv()
        if not argv:
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "Rerun 播放",
                "未找到 Rerun。请安装 rerun-sdk，或设置 RERUN_BIN。\n"
                "例如: pip install rerun-sdk",
            )
            return False
        cmd = list(argv) + [path]
        ok = _eai_platform_bootstrap.QProcess.startDetached(cmd[0], cmd[1:])
        if not ok:
            quoted = " ".join(_eai_platform_bootstrap.shlex.quote(x) for x in cmd)
            ok = _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", quoted])
        if ok:
            self.status_bar.showMessage(
                f"已启动 rerun-sdk: {_eai_platform_bootstrap.os.path.basename(path)}",
                6000,
            )
            return True
        _eai_platform_bootstrap.QMessageBox.warning(
            self,
            "Rerun 播放",
            "启动 Rerun 失败：\n" + " ".join(cmd),
        )
        return False

    def _stop_ctx_rerun_embed(self) -> None:
        # 兼容关闭窗口时清理；原生 Viewer 为独立进程，此处仅复位面板文案
        if getattr(self, "ctx_rerun_title", None) is not None:
            self.ctx_rerun_title.setText("Rerun（rerun-sdk）")
        if getattr(self, "ctx_rerun_placeholder", None) is not None:
            self.ctx_rerun_placeholder.setText(
                "选择 .rrd 后点「Rerun 播放」，\n"
                "将用 rerun-sdk 原生 Viewer 打开该文件。\n\n"
                "等价命令: rerun <file.rrd>"
            )

    def _install_scroll_on_all_tabs(self) -> int:
        tabs_list = []
        root_tabs = getattr(self, 'control_tabs', None)
        if isinstance(root_tabs, _eai_platform_bootstrap.QTabWidget):
            tabs_list.append(root_tabs)
        tabs_list.extend(self.findChildren(_eai_platform_bootstrap.QTabWidget))
        unique = []
        for tabs in tabs_list:
            if tabs not in unique:
                unique.append(tabs)
        # Deepest first so nested pages are configured before their parent page
        # is placed inside the outer scroll area.
        unique.sort(key=lambda widget: len(widget.findChildren(_eai_platform_bootstrap.QTabWidget)))
        return sum(_eai_ui_widgets_layouts._install_scroll_for_tab_widget(tabs) for tabs in unique)

    def _apply_only_tabs(self, only_tabs: _eai_platform_bootstrap.Sequence[str]) -> None:
        """启动时只保留指定控制区 tab；单个时隐藏 tab 栏。"""
        tabs = getattr(self, "control_tabs", None)
        if tabs is None or not only_tabs:
            return
        wanted = {str(t) for t in only_tabs}
        # 优先按用户保存的顺序，再补默认顺序
        order = _eai_infrastructure_ui_storage.preferred_control_tabs_order(
            [t for t in _eai_configuration_tabs.CONTROL_TAB_TITLES if t in wanted]
        )
        for i in range(tabs.count() - 1, -1, -1):
            title = tabs.tabText(i)
            if title not in wanted:
                tabs.removeTab(i)
        # 按目标顺序重排
        if order and tabs.count() > 1:
            widgets: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.QWidget]] = []
            for i in range(tabs.count()):
                widgets.append((tabs.tabText(i), tabs.widget(i)))
            by_title = {t: w for t, w in widgets}
            while tabs.count():
                tabs.removeTab(0)
            for title in order:
                w = by_title.get(title)
                if w is not None:
                    tabs.addTab(w, title)
        status_bar = getattr(self, "status_bar", None)
        if tabs.count() == 0:
            if status_bar is not None:
                status_bar.showMessage("警告: --tab 过滤后无可用标签页")
            return
        tabs.setCurrentIndex(0)
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        self._refresh_control_tab_tooltips()
        if tabs.count() == 1:
            tabs.tabBar().setVisible(False)
            self.setWindowTitle(f"Camera Topic Viewer — {titles[0]}")
            if status_bar is not None:
                status_bar.showMessage(f"仅显示 tab: {titles[0]}")
        else:
            tabs.tabBar().setVisible(True)
            joined = " / ".join(titles)
            self.setWindowTitle(f"Camera Topic Viewer — {joined}")
            if status_bar is not None:
                status_bar.showMessage(f"显示 tabs: {joined}")

    def _misc_child_titles(self) -> _eai_platform_bootstrap.List[str]:
        inner = getattr(self, "misc_inner_tabs", None)
        if inner is None:
            return []
        return [inner.tabText(i) for i in range(inner.count())]

    def _update_misc_empty_hint(self) -> None:
        hint = getattr(self, "_misc_empty_hint", None)
        inner = getattr(self, "misc_inner_tabs", None)
        if hint is None or inner is None:
            return
        hint.setVisible(inner.count() == 0)

    def _index_of_control_tab(self, title: str) -> int:
        tabs = getattr(self, "control_tabs", None)
        if tabs is None:
            return -1
        for i in range(tabs.count()):
            if tabs.tabText(i) == title:
                return i
        return -1

    def _index_of_misc_child(self, title: str) -> int:
        inner = getattr(self, "misc_inner_tabs", None)
        if inner is None:
            return -1
        for i in range(inner.count()):
            if inner.tabText(i) == title:
                return i
        return -1

    def _current_control_tab_title(self) -> str:
        tabs = getattr(self, "control_tabs", None)
        if tabs is None or tabs.count() <= 0:
            return ""
        title = str(tabs.tabText(tabs.currentIndex()) or "").strip()
        if title != _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE:
            return title
        inner = getattr(self, "misc_inner_tabs", None)
        if inner is not None and inner.count() > 0:
            child = str(inner.tabText(inner.currentIndex()) or "").strip()
            if child:
                return child
        return title

    def _pause_ui_frames_for_interaction(self, seconds: float = 1.5) -> None:
        """切 tab / 点击时暂停刷图，把主线程让给 Qt 事件。"""
        until = _eai_platform_bootstrap.time.time() + float(seconds)
        self._ui_frame_paused_until = until
        try:
            self.ros_bridge._ui_frame_paused_until = until
        except Exception:
            pass
        try:
            self.node.take_pending_ui_frames()
        except Exception:
            pass

