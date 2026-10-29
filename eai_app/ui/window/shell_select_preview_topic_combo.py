"""ui.window.shell select preview topic combo — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ShellSelectPreviewTopicComboMixin:
    def _select_preview_topic_combo(
        self, combo: object, topic: str, *, status_cb=None
    ) -> None:
        if combo is None or not topic:
            return
        if _eai_perception_image_geometry.is_depth_topic(topic):
            color_topic = _eai_perception_image_geometry.find_paired_color_topic(topic, self._cached_topic_names())
            if not color_topic:
                return
            topic = color_topic
        self._refresh_preview_topic_combo(combo, prefer=topic)
        idx = combo.findData(topic)
        if idx < 0:
            combo.addItem(topic, topic)
            idx = combo.findData(topic)
        if idx >= 0 and combo.currentIndex() != idx:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        self._last_sam3_topic = topic
        if status_cb is not None:
            short = topic.rsplit("/", 1)[-1]
            status_cb(f"已选预览: {short}")

    def _update_lingbot_depth_pair_label(self) -> None:
        if not hasattr(self, "lingbot_depth_pair_label"):
            return
        source = str(self.lingbot_depth_source_combo.currentData() or "camera")
        if source != "camera":
            self.lingbot_depth_pair_label.setText("配对深度: —")
            return
        topic = str(self.lingbot_depth_topic_combo.currentData() or "").strip()
        if not topic:
            self.lingbot_depth_pair_label.setText("配对深度: （未选预览）")
            return
        depth_topic = self._get_paired_depth_topic_name(topic)
        if not depth_topic:
            self.lingbot_depth_pair_label.setText("配对深度: 无")
            self.lingbot_depth_pair_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_RED};")
            return
        has_frame = self._get_paired_depth_frame(topic) is not None
        if not has_frame:
            depth_panel = self.panels.get(depth_topic)
            if (
                isinstance(depth_panel, _eai_ui_cameras_panels.DepthPanel3D)
                and depth_panel._latest_depth is not None
            ):
                has_frame = True
        enabled = self._is_topic_enabled(depth_topic)
        if has_frame:
            status = "有帧"
            color = _eai_configuration_policies.UI_ACCENT_GREEN
        elif enabled:
            status = "等待帧"
            color = _eai_configuration_policies.UI_TEXT_MUTED
        else:
            status = "未勾选"
            color = _eai_configuration_policies.UI_ACCENT_RED
        short = depth_topic.rsplit("/", 1)[-1]
        self.lingbot_depth_pair_label.setText(f"配对深度: {short} ({status})")
        self.lingbot_depth_pair_label.setStyleSheet(f"color: {color};")

    def _select_lingbot_depth_topic(
        self, topic: str, *, from_click: bool = False
    ) -> None:
        if not hasattr(self, "lingbot_depth_topic_combo"):
            return
        self._select_preview_topic_combo(
            self.lingbot_depth_topic_combo,
            topic,
            status_cb=self._on_lingbot_depth_status if from_click else None,
        )
        self._update_lingbot_depth_pair_label()

    def _refresh_lingbot_depth_topic_combo(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "lingbot_depth_topic_combo"):
            return
        self._refresh_preview_topic_combo(self.lingbot_depth_topic_combo, prefer=prefer)
        self._update_lingbot_depth_pair_label()

    def _refresh_lingbot_video_topic_combo(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "lingbot_video_topic_combo"):
            return
        self._refresh_preview_topic_combo(self.lingbot_video_topic_combo, prefer=prefer)

    def _refresh_lingbot_world_topic_combo(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "lingbot_world_topic_combo"):
            return
        self._refresh_preview_topic_combo(self.lingbot_world_topic_combo, prefer=prefer)

    def _load_sim_bridge_depth_u16(
        self,
        color_topic: str,
        *,
        color_bgr: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        max_age_behind_color_s: float = 2.0,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]:
        """直读仿真共享目录 {cam_key}_depth.npy（不依赖 ROS topic 发现）。

        若 depth 明显比同相机 color 旧（例如 LIBERO 只写 RGB、残留 Isaac depth），
        或与彩色分辨率差过大，则丢弃，避免 6D 估计崩溃。
        """
        cam_key = self._sim_bridge_cam_key_for_color(color_topic)
        if not cam_key:
            return None
        path = _eai_platform_bootstrap.os.path.join(self._sim_bridge_dir(), f"{cam_key}_depth.npy")
        if not _eai_platform_bootstrap.os.path.isfile(path):
            return None
        color_mtime = self._sim_bridge_npy_mtime(cam_key, depth=False)
        depth_mtime = self._sim_bridge_npy_mtime(cam_key, depth=True)
        if (
            color_mtime is not None
            and depth_mtime is not None
            and (color_mtime - depth_mtime) > float(max_age_behind_color_s)
        ):
            return None
        try:
            with open(path, "rb") as f:
                arr = _eai_platform_bootstrap.np.load(f)
            arr = _eai_platform_bootstrap.np.asarray(arr)
        except Exception:
            return None
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None
        if color_bgr is not None:
            ch, cw = _eai_platform_bootstrap.np.asarray(color_bgr).shape[:2]
            dh, dw = arr.shape[:2]
            # 分辨率差过大 → 视为不同会话残留帧
            if ch > 0 and cw > 0 and dh > 0 and dw > 0:
                scale = max(float(ch) / float(dh), float(cw) / float(dw),
                            float(dh) / float(ch), float(dw) / float(cw))
                if scale > 1.8:
                    return None
        if arr.dtype == _eai_platform_bootstrap.np.uint16:
            return _eai_platform_bootstrap.np.ascontiguousarray(arr)
        depth_f = arr.astype(_eai_platform_bootstrap.np.float32)
        finite = depth_f[_eai_platform_bootstrap.np.isfinite(depth_f)]
        if finite.size and float(_eai_platform_bootstrap.np.nanmax(finite)) < 100.0:
            out = _eai_platform_bootstrap.np.clip(depth_f * 1000.0, 0, 65535).astype(_eai_platform_bootstrap.np.uint16)
        else:
            out = _eai_platform_bootstrap.np.clip(depth_f, 0, 65535).astype(_eai_platform_bootstrap.np.uint16)
        return _eai_platform_bootstrap.np.ascontiguousarray(out)

    def _resolve_lingbot_depth_inputs(
        self,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, str, str, str]]:
        """返回 (example_id, rgb, depth, intrinsics)；example 模式时后三者为空。"""
        self._lingbot_depth_capture_topic = ""
        source = str(self.lingbot_depth_source_combo.currentData() or "camera")
        if source == "example":
            return str(int(self.lingbot_depth_example_spin.value())), "", "", ""

        if source == "camera":
            self._refresh_lingbot_depth_topic_combo()
            color_source = self._pick_lingbot_depth_color_source()
            if color_source is None:
                self._append_lingbot_depth_log(
                    "[ERROR] 无可用彩色图像：请在「预览图像」中选择，"
                    "或点击图像预览，或勾选 color topic"
                )
                self._on_lingbot_depth_status("无可用相机图像")
                return None
            color_topic, color_bgr, _panel = color_source
            depth_topic = self._get_paired_depth_topic_name(color_topic)
            if not depth_topic:
                self._append_lingbot_depth_log(
                    f"[ERROR] {color_topic} 无配对 depth topic"
                    "（约定应为 *_depth，如 /camera/head_depth）"
                )
                self._on_lingbot_depth_status("无配对深度 topic")
                return None

            if self._ensure_topic_enabled_for_capture(depth_topic):
                self._append_lingbot_depth_log(
                    f"已确保配对深度已勾选/订阅: {depth_topic}"
                )
            elif depth_topic not in self.topic_checks:
                self._append_lingbot_depth_log(
                    f"[WARN] topic 列表中尚无 {depth_topic}，"
                    "将尝试直读仿真共享 depth .npy"
                )

            depth = self._get_paired_depth_frame(color_topic)
            if depth is None:
                # 刚勾选时等一小会儿让 ROS 帧进来
                app = _eai_platform_bootstrap.QApplication.instance()
                for _ in range(20):
                    if app is not None:
                        app.processEvents()
                    depth = self._get_paired_depth_frame(color_topic)
                    if depth is not None:
                        break
                    _eai_platform_bootstrap.time.sleep(0.05)

            if depth is None and depth_topic:
                depth_panel = self.panels.get(depth_topic)
                if (
                    isinstance(depth_panel, _eai_ui_cameras_panels.DepthPanel3D)
                    and depth_panel._latest_depth is not None
                ):
                    depth = _eai_platform_bootstrap.np.asarray(depth_panel._latest_depth)

            if depth is None:
                depth = self._load_sim_bridge_depth_u16(color_topic)
                if depth is not None:
                    self._frame_cache[depth_topic] = depth
                    self._append_lingbot_depth_log(
                        f"已从仿真共享目录读取深度: "
                        f"{self._sim_bridge_cam_key_for_color(color_topic)}_depth.npy"
                    )

            if depth is None:
                bridge_dir = self._sim_bridge_dir()
                cam_key = self._sim_bridge_cam_key_for_color(color_topic) or "?"
                depth_npy = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{cam_key}_depth.npy")
                if depth_topic not in self.topic_checks and not _eai_platform_bootstrap.os.path.isfile(depth_npy):
                    self._append_lingbot_depth_log(
                        f"[ERROR] 未发现 {depth_topic}，且共享目录无 {cam_key}_depth.npy。\n"
                        f"  请重启仿真评测（需 arx_x5 approximate_depth=true + "
                        f"camera_config distance_to_image_plane），"
                        f"或改用「官方示例」/「本地文件」。\n"
                        f"  检查: {depth_npy}"
                    )
                    self._on_lingbot_depth_status("无仿真深度帧")
                elif not self._is_topic_enabled(depth_topic) and depth_topic in self.topic_checks:
                    self._append_lingbot_depth_log(
                        f"[ERROR] 请勾选配对深度 topic: {depth_topic}"
                    )
                    self._on_lingbot_depth_status("请勾选配对深度")
                else:
                    self._append_lingbot_depth_log(
                        f"[ERROR] 无法读取配对深度帧: {depth_topic}"
                    )
                    self._on_lingbot_depth_status("无深度帧")
                return None

            from eai_app.perception.depth_io import write_camera_inputs
            import tempfile
            _eai_platform_bootstrap.os.makedirs(_eai_configuration_tabs.LINGBOT_DEPTH_CACHE_DIR, exist_ok=True)
            out_dir = tempfile.mkdtemp(prefix="camera_input_", dir=_eai_configuration_tabs.LINGBOT_DEPTH_CACHE_DIR)
            try:
                bgr = _eai_platform_bootstrap.np.array(color_bgr, copy=True)
                depth_arr = _eai_platform_bootstrap.np.array(depth, copy=True)
                if bgr.ndim == 2:
                    bgr = _eai_platform_bootstrap.cv2.cvtColor(bgr, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
                if depth_arr.ndim != 2:
                    raise ValueError("相机深度必须是二维数组")
                h, w = depth_arr.shape
                fx, fy, cx, cy = self.node.get_intrinsics(depth_topic, w, h)
                K = _eai_platform_bootstrap.np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=_eai_platform_bootstrap.np.float64)
                rgb_path, depth_path, K_path = write_camera_inputs(out_dir, bgr, depth_arr, K)
                self._lingbot_depth_capture_topic = color_topic
            except (ValueError, OSError) as exc:
                self._append_lingbot_depth_log(f"[ERROR] {exc}")
                self._on_lingbot_depth_status(f"输入无效: {exc}")
                return None
            self._append_lingbot_depth_log(
                f"已抓帧 color={color_topic} depth={depth_topic} -> {out_dir}"
            )
            return "", rgb_path, depth_path, K_path

        rgb = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(self.lingbot_depth_rgb_edit.text().strip())
        )
        depth = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(self.lingbot_depth_depth_edit.text().strip())
        )
        intrinsics = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(self.lingbot_depth_intrinsics_edit.text().strip())
        )
        for label, path in (("RGB", rgb), ("Depth", depth), ("Intrinsics", intrinsics)):
            if not _eai_platform_bootstrap.os.path.isfile(path):
                self._append_lingbot_depth_log(f"[ERROR] {label} 不存在: {path}")
                self._on_lingbot_depth_status(f"{label} 不存在")
                return None
        return "", rgb, depth, intrinsics

    def _clear_lingbot_depth_result(self) -> None:
        self._robot_lingbot_depth = None
        view = getattr(self, "_robot_view", None)
        if view is not None and hasattr(view, "set_lingbot_depth"):
            view.set_lingbot_depth(None)
        label = getattr(self, "lingbot_depth_refined_label", None)
        if label is not None:
            label.set_source_pixmap(None)

    def _show_lingbot_depth_on_robot_view(
        self, result: dict, *, silent: bool = False
    ) -> None:
        ply = str(result.get("point_cloud") or "").strip()
        if not ply or not _eai_platform_bootstrap.os.path.isfile(ply):
            self._clear_lingbot_depth_result()
            if not silent:
                self.status_bar.showMessage(
                    "LingBot-Depth 没有 point_cloud.ply，机器人页未更新。", 8000
                )
            return
        color_topic = str(
            result["color_topic"] if "color_topic" in result
            else getattr(self, "_lingbot_depth_last_color_topic", "") or ""
        ).strip()
        if not color_topic:
            self._robot_lingbot_depth = None
            view = getattr(self, "_robot_view", None)
            if view is not None and hasattr(view, "set_lingbot_depth"):
                view.set_lingbot_depth(None)
            if not silent:
                self.status_bar.showMessage("示例/本地文件结果已生成；未绑定实机相机，不放入机器人坐标系。", 8000)
            return
        payload = {
            "ply": ply,
            "color_topic": color_topic,
            "points": int(result.get("points") or 0),
        }
        self._robot_lingbot_depth = payload
        view = getattr(self, "_robot_view", None)
        if view is None or not hasattr(view, "set_lingbot_depth_from_ply"):
            if not silent:
                self.status_bar.showMessage(
                    "LingBot-Depth 点云已缓存；打开「手臂/手」→「机器人」查看。",
                    6000,
                )
            return
        try:
            view.set_lingbot_depth_from_ply(ply, color_topic=color_topic)
        except Exception as exc:
            self._clear_lingbot_depth_result()
            if not silent:
                self.status_bar.showMessage(
                    f"机器人页未画出 LingBot-Depth：{exc}", 8000
                )
            return
        if not silent:
            self.status_bar.showMessage(
                f"已在机器人页显示 LingBot-Depth（{payload['points']} 点）", 5000
            )

    def _append_reward_log(self, line: str) -> None:
        if not hasattr(self, "reward_log_edit"):
            return
        self.reward_log_edit.append(line)
        bar = self.reward_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_reward_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_reward_launcher", None)
            and self._reward_launcher.is_running()
        )
        if hasattr(self, "reward_run_btn"):
            self.reward_run_btn.setEnabled(not running)
        if hasattr(self, "reward_stop_btn"):
            self.reward_stop_btn.setEnabled(running)
        for w in (
            getattr(self, "reward_mode_combo", None),
            getattr(self, "reward_type_combo", None),
            getattr(self, "reward_arch_combo", None),
            getattr(self, "reward_ckpt_edit", None),
            getattr(self, "reward_root_edit", None),
        ):
            if w is not None:
                w.setEnabled(not running)

    def _refresh_reward_topic_combo(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "reward_topic_combo"):
            return
        self._refresh_preview_topic_combo(self.reward_topic_combo, prefer=prefer)

    def _append_reward_wf_log(self, line: str) -> None:
        if not hasattr(self, "reward_wf_log_edit"):
            return
        self.reward_wf_log_edit.append(line)
        bar = self.reward_wf_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_reward_wf_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_reward_wf_launcher", None)
            and self._reward_wf_launcher.is_running()
        )
        if hasattr(self, "reward_wf_start_btn"):
            self.reward_wf_start_btn.setEnabled(not running)
        if hasattr(self, "reward_wf_stop_btn"):
            self.reward_wf_stop_btn.setEnabled(running)
        for wname in (
            "reward_wf_mode_combo",
            "reward_wf_dataset_combo",
            "reward_wf_config_combo",
            "reward_wf_root_edit",
            "reward_wf_python_edit",
            "reward_wf_cuda_edit",
            "reward_wf_overrides_edit",
            "reward_wf_refresh_btn",
            "reward_wf_raw_edit",
            "reward_wf_out_edit",
            "reward_wf_train_pt_edit",
            "reward_wf_val_pt_edit",
            "reward_wf_samples_spin",
            "reward_wf_val_spin",
            "reward_wf_ratio_spin",
            "reward_wf_seed_spin",
            "reward_wf_keep_last_check",
            "reward_wf_root_browse_btn",
            "reward_wf_raw_browse_btn",
            "reward_wf_out_browse_btn",
            "reward_wf_train_pt_browse_btn",
            "reward_wf_val_pt_browse_btn",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                w.setEnabled(not running)

    def _refresh_reward_wf_configs(self, prefer: _eai_platform_bootstrap.Optional[str] = None) -> None:
        if not hasattr(self, "reward_wf_config_combo"):
            return
        root = self.reward_wf_root_edit.text().strip()
        mode = str(self.reward_wf_mode_combo.currentData() or "preprocess")
        kind = "collect" if mode == "collect" else "train"
        names = _eai_adapters_models_discovery.list_reward_workflow_configs(root, kind=kind)
        if mode == "preprocess":
            # 预处理不依赖 hydra config，仍展示训练 config 便于 pipeline 切换
            names = _eai_adapters_models_discovery.list_reward_workflow_configs(root, kind="train") or [
                "reward_training"
            ]
        current = prefer or self.reward_wf_config_combo.currentText().strip()
        if not current:
            current = (
                "realworld_collect_dataset"
                if mode == "collect"
                else "reward_training"
            )
        self.reward_wf_config_combo.blockSignals(True)
        self.reward_wf_config_combo.clear()
        for name in names:
            self.reward_wf_config_combo.addItem(name, name)
        if not names:
            self.reward_wf_config_combo.addItem(current, current)
        idx = self.reward_wf_config_combo.findData(current)
        if idx < 0:
            idx = self.reward_wf_config_combo.findText(current)
        if idx >= 0:
            self.reward_wf_config_combo.setCurrentIndex(idx)
        else:
            self.reward_wf_config_combo.setEditText(current)
        self.reward_wf_config_combo.blockSignals(False)

    def _append_mujoco_log(self, line: str) -> None:
        if not hasattr(self, "mujoco_log_edit"):
            return
        if _eai_adapters_models_discovery.rynnvalue_status_log_is_routine(line):
            return
        self.mujoco_log_edit.append(line)
        bar = self.mujoco_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _is_spaces_sim_backend(self, backend: str = "") -> bool:
        """MolmoSpaces or its Psi fork (share the same stack page / launcher)."""
        return (backend or self._sim_backend_id()) in (
            "molmospaces",
            "psi_molmospaces",
        )

    def _is_extended_sim_backend(self, backend: str = "") -> bool:
        key = backend or self._sim_backend_id()
        if key in (
            "isaac",
            "mujoco",
            "molmospaces",
            "psi_molmospaces",
            "arena",
            "libero",
            "robotwin",
        ):
            return False
        # extended radio selected, or known extended key
        if (
            getattr(self, "sim_backend_extended_radio", None) is not None
            and self.sim_backend_extended_radio.isChecked()
        ):
            return True
        spec = _eai_configuration_models.get_sim_backend_spec(key)
        return bool(spec and getattr(spec, "kind", "") == "extended")

    def _apply_sim_leaderboard_board(self, board: _eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any]) -> None:
        """Render one board dict into the shared leaderboard widgets."""
        if not hasattr(self, "sim_leaderboard_table") or not isinstance(board, dict):
            return
        title = str(board.get("title") or "业内评测榜单")
        self.sim_leaderboard_group.setTitle(title)
        metric = str(board.get("metric") or "")
        protocol = str(board.get("protocol") or "")
        updated = str(
            board.get("_fetched_at")
            or board.get("last_live_fetch")
            or board.get("_updated")
            or ""
        )
        source_label = str(board.get("source_label") or "来源")
        n_ind = board.get("_industry_count")
        n_loc = board.get("_local_count")
        if n_ind is None or n_loc is None:
            raw_entries = board.get("entries") or []
            n_loc = sum(
                1
                for e in raw_entries
                if isinstance(e, dict) and str(e.get("origin") or "") == "local"
            )
            n_ind = max(0, len([e for e in raw_entries if isinstance(e, dict)]) - int(n_loc))
        meta_bits = [b for b in (metric, protocol) if b]
        if board.get("_live") or board.get("_fetched_at"):
            meta_bits.append(f"实时 {updated}" if updated else "实时")
        elif updated:
            meta_bits.append(f"快照 {updated}")
        if int(n_loc or 0) > 0:
            meta_bits.append(f"业内 {int(n_ind)} + 本地 {int(n_loc)}")
        self.sim_leaderboard_meta_label.setText(" · ".join(meta_bits))
        self._sim_leaderboard_source_url = str(board.get("source_url") or "").strip()
        self.sim_leaderboard_source_btn.setEnabled(bool(self._sim_leaderboard_source_url))
        self.sim_leaderboard_source_btn.setText("打开来源")
        tip_url = self._sim_leaderboard_source_url or "无来源链接"
        self.sim_leaderboard_source_btn.setToolTip(f"{source_label}\n{tip_url}")

        # Prefer pre-combined entries; otherwise combine industry + local_entries.
        entries = [e for e in (board.get("entries") or []) if isinstance(e, dict)]
        if not board.get("_combined"):
            local_rows = [
                e for e in (board.get("local_entries") or []) if isinstance(e, dict)
            ]
            if local_rows:
                try:
                    from eai_app.ui.simulation.leaderboard import combine_board_entries

                    industry = [
                        e
                        for e in entries
                        if str(e.get("origin") or "") != "local"
                    ]
                    entries = combine_board_entries(industry, local_rows)
                    n_ind = len(industry)
                    n_loc = len(local_rows)
                except Exception:
                    pass

        self._sim_leaderboard_row_meta = []
        self.sim_leaderboard_table.setRowCount(len(entries))
        local_bg = _eai_platform_bootstrap.QColor("#1e3a28")
        local_fg = _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_ACCENT_GREEN)
        for row, entry in enumerate(entries):
            origin = str(entry.get("origin") or "").strip().lower()
            is_local = origin == "local" or bool(entry.get("local_id"))
            origin_label = "本地" if is_local else "业内"
            cells = [
                str(entry.get("rank", row + 1)),
                origin_label,
                str(entry.get("method") or ""),
                str(entry.get("score") or ""),
                str(entry.get("detail") or ""),
                str(entry.get("org") or ""),
                str(entry.get("date") or ""),
            ]
            tip_bits = [
                t
                for t in (
                    str(entry.get("note") or "").strip(),
                    str(entry.get("result_path") or "").strip(),
                    str(entry.get("local_id") or "").strip() if is_local else "",
                )
                if t
            ]
            tip = "\n".join(tip_bits)
            self._sim_leaderboard_row_meta.append(
                {
                    "origin": "local" if is_local else "industry",
                    "local_id": str(entry.get("local_id") or ""),
                    "method": str(entry.get("method") or ""),
                }
            )
            for col, text in enumerate(cells):
                item = _eai_platform_bootstrap.QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                if col in (0, 1):
                    item.setTextAlignment(int(_eai_platform_bootstrap.Qt.AlignCenter))
                if is_local:
                    item.setBackground(local_bg)
                    if col == 1:
                        item.setForeground(local_fg)
                self.sim_leaderboard_table.setItem(row, col, item)
        notes = []
        if board.get("_global_note"):
            notes.append(str(board["_global_note"]))
        if board.get("_live") or board.get("_fetched_at"):
            notes.append(
                "已从公开源实时更新"
                + (f"（{board.get('_fetched_at')}）" if board.get("_fetched_at") else "")
                + "；本地评测行已保留并重新排名"
            )
        if int(n_loc or 0) > 0:
            notes.append(
                f"对比视图：业内 {int(n_ind)} 条 + 本地 {int(n_loc)} 条（绿色「本地」行）"
            )
        local_notes = [
            f"{e.get('method')}: {e.get('score')}"
            for e in entries
            if str(e.get("origin") or "") == "local" or e.get("local_id")
        ]
        if local_notes:
            notes.append("本地 · " + "；".join(local_notes[:6]))
        self.sim_leaderboard_note_label.setText("\n".join(notes))

    def _refresh_sim_leaderboard(self, backend: str = "") -> None:
        """Fill the shared industry+local leaderboard for the current sim backend."""
        if not hasattr(self, "sim_leaderboard_table"):
            return
        key = (backend or self._sim_backend_id() or "isaac").strip().lower()
        board = None
        try:
            board = _eai_configuration_models.load_sim_eval_board_display(key)
            if not board:
                board = _eai_configuration_models.load_sim_eval_board(key)
        except Exception as exc:
            self.sim_leaderboard_meta_label.setText(f"榜单加载失败: {exc}")
            self.sim_leaderboard_table.setRowCount(0)
            self.sim_leaderboard_note_label.setText("")
            self._sim_leaderboard_source_url = ""
            self.sim_leaderboard_source_btn.setEnabled(False)
            self._sim_leaderboard_row_meta = []
            return

        if not board:
            self.sim_leaderboard_group.setTitle("业内评测榜单")
            self.sim_leaderboard_meta_label.setText(
                f"暂无「{key}」后端的业内结果快照。"
            )
            self.sim_leaderboard_table.setRowCount(0)
            self.sim_leaderboard_note_label.setText("")
            self._sim_leaderboard_source_url = ""
            self.sim_leaderboard_source_btn.setEnabled(False)
            self._sim_leaderboard_row_meta = []
            return

        self._apply_sim_leaderboard_board(board)

    def _refresh_ext_bench_form(self) -> None:
        """Sync extended-page fields from the selected suite."""
        if not hasattr(self, "ext_bench_root_edit"):
            return
        key = "simpler"
        if hasattr(self, "sim_extended_combo"):
            key = str(self.sim_extended_combo.currentData() or "simpler").strip()
        key = _eai_configuration_models.normalize_sim_backend_key(key)
        spec = _eai_configuration_models.get_sim_backend_spec(key)
        label = getattr(spec, "label", None) or key
        if hasattr(self, "ext_bench_title_label"):
            self.ext_bench_title_label.setText(f"扩展评测 · {label}")
        if hasattr(self, "ext_bench_hint_label"):
            hint = getattr(spec, "hint", "") or ""
            engine = getattr(spec, "engine", "") or ""
            if engine:
                hint = f"{hint}  引擎: {engine}".strip()
            self.ext_bench_hint_label.setText(hint)
        root = ""
        if spec is not None:
            try:
                root = _eai_configuration_models.resolve_extended_default_root(spec) or ""
            except Exception:
                roots = getattr(spec, "default_roots", ()) or ()
                root = roots[0] if roots else ""
        # Don't clobber a user-edited path for the same backend.
        cur = self.ext_bench_root_edit.text().strip()
        prev_key = getattr(self, "_ext_bench_form_key", "")
        if key != prev_key or not cur:
            self.ext_bench_root_edit.setText(root)
        self._ext_bench_form_key = key
        if hasattr(self, "ext_bench_status_label"):
            self.ext_bench_status_label.setText(f"空闲 · {label}")
        self._refresh_ext_task_thumb()

    def _refresh_ext_task_thumb(self, *_args) -> None:
        thumb = getattr(self, "ext_task_thumb", None)
        if thumb is None:
            return
        suite = ""
        if hasattr(self, "sim_extended_combo"):
            suite = str(self.sim_extended_combo.currentText() or "").strip()
        task = ""
        if hasattr(self, "ext_bench_task_edit"):
            task = self.ext_bench_task_edit.text().strip()
        thumb.show_text(f"{suite} {task}".strip())

    def _update_sim_backend_mutex_ui(self, *_args) -> None:
        """Cross-disable Isaac vs MuJoCo/MolmoSpaces while either side is running."""
        mj_running = bool(
            getattr(self, "_mujoco_launcher", None)
            and self._mujoco_launcher.is_running()
        )
        isaac_running = bool(
            (
                getattr(self, "_sim_eval_launcher", None)
                and self._sim_eval_launcher.is_running()
            )
            or (
                getattr(self, "_sim_bridge_launcher", None)
                and self._sim_bridge_launcher.is_running()
            )
        )
        busy = bool(mj_running or isaac_running)
        for wname in (
            "sim_backend_tab_bar",
            "sim_backend_isaac_radio",
            "sim_backend_mujoco_radio",
            "sim_backend_spaces_radio",
            "sim_backend_psi_spaces_radio",
            "sim_backend_arena_radio",
            "sim_backend_libero_radio",
            "sim_backend_robotwin_radio",
            "sim_backend_extended_radio",
            "sim_extended_combo",
        ):
            w = getattr(self, wname, None)
            if w is not None:
                # Keep extended combo enabled only when extended radio is selected
                # and nothing is running.
                if wname == "sim_extended_combo":
                    w.setEnabled(
                        (not busy)
                        and bool(
                            getattr(self, "sim_backend_extended_radio", None)
                            and self.sim_backend_extended_radio.isChecked()
                        )
                    )
                else:
                    w.setEnabled(not busy)

        backend = self._sim_backend_id()
        mj_startable = (
            (not mj_running)
            and (not isaac_running)
            and (
                backend
                in (
                    "mujoco",
                    "molmospaces",
                    "psi_molmospaces",
                    "arena",
                    "libero",
                    "robotwin",
                )
                or self._is_extended_sim_backend(backend)
            )
        )
        if hasattr(self, "mujoco_start_btn"):
            self.mujoco_start_btn.setEnabled(mj_startable)
        if hasattr(self, "mujoco_viewer_only_btn"):
            self.mujoco_viewer_only_btn.setEnabled(mj_startable)

        if mj_running:
            if hasattr(self, "sim_eval_start_btn"):
                self.sim_eval_start_btn.setEnabled(False)
            if hasattr(self, "sim_eval_viewer_only_btn"):
                self.sim_eval_viewer_only_btn.setEnabled(False)
            if hasattr(self, "sim_bridge_start_btn"):
                self.sim_bridge_start_btn.setEnabled(False)
            return

        # Isaac side idle/running state restored by its own UI updaters.
        if hasattr(self, "_update_sim_eval_run_ui"):
            try:
                self._update_sim_eval_run_ui()
            except Exception:
                pass
        if hasattr(self, "_update_sim_bridge_ui"):
            try:
                self._update_sim_bridge_ui()
            except Exception:
                pass

