"""ui.window.shell inject bagel image — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_bagel_runtime = module_ref('eai_app.adapters.models.bagel_runtime')
_eai_adapters_models_discovery = module_ref('eai_app.adapters.models.discovery')
_eai_adapters_simulation_robodojo = module_ref('eai_app.adapters.simulation.robodojo')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ShellInjectBagelImageMixin:
    def _inject_bagel_image(self, path: str) -> None:
        """把本地图片以内存 File 注入嵌入 Gradio，避免 WebEngine 空上传。"""
        img = _eai_platform_bootstrap.QImage(path)
        if img.isNull():
            raise ValueError(f"无法读取图像: {path}")
        max_edge = 2048
        if img.width() > max_edge or img.height() > max_edge:
            img = img.scaled(
                max_edge,
                max_edge,
                _eai_platform_bootstrap.Qt.KeepAspectRatio,
                _eai_platform_bootstrap.Qt.SmoothTransformation,
            )
        buf = _eai_platform_bootstrap.QByteArray()
        qbuf = _eai_platform_bootstrap.QBuffer(buf)
        qbuf.open(_eai_platform_bootstrap.QIODevice.WriteOnly)
        if not img.save(qbuf, "PNG"):
            raise RuntimeError("PNG 编码失败")
        qbuf.close()
        b64 = bytes(buf.toBase64()).decode("ascii")
        name = _eai_platform_bootstrap.os.path.splitext(_eai_platform_bootstrap.os.path.basename(path))[0] + ".png"
        # JSON-encode args for safe JS string literals
        payload = _eai_platform_bootstrap.json.dumps([b64, "image/png", name], ensure_ascii=True)
        js = f"({_eai_adapters_models_bagel_runtime._BAGEL_INJECT_IMAGE_JS})(...{payload})"

        def _on_js_result(result) -> None:
            text = str(result) if result is not None else ""
            if text.startswith("ok:"):
                self._append_bagel_log(f"[ok] 已注入图片 → Gradio ({name})")
                self.status_bar.showMessage(f"已写入 Gradio: {name}")
            else:
                self._append_bagel_log(f"[warn] 图片注入结果: {text or 'empty'}")
                self.status_bar.showMessage(
                    "注入可能未成功：请先切到 Image Edit / Understanding 标签再试"
                )

        page = self.bagel_web_view.page()
        try:
            page.runJavaScript(js, _on_js_result)
        except TypeError:
            # 部分 PyQt 绑定不接受 callback
            page.runJavaScript(js)
            self._append_bagel_log(f"[ok] 已请求注入图片 → Gradio ({name})")
            self.status_bar.showMessage(f"已写入 Gradio: {name}")

    def _append_sim_log(self, line: str) -> None:
        if _eai_adapters_models_discovery.rynnvalue_status_log_is_routine(line):
            return
        self.sim_log_edit.append(line)
        scrollbar = self.sim_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_sim_bridge_ui(self, *_args) -> None:
        running = self._sim_bridge_launcher.is_running()
        self.sim_bridge_stop_btn.setEnabled(running)
        self.sim_bridge_start_btn.setEnabled(not running)
        self.sim_bridge_dir_edit.setEnabled(not running)
        self.sim_bridge_browse_btn.setEnabled(not running)
        self.sim_bridge_hz_spin.setEnabled(not running)
        if running:
            self.sim_status_label.setText("桥: 运行中")
            self.sim_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.sim_status_label.setText("桥: 空闲")
            self.sim_status_label.setStyleSheet("")
        # 帧状态 + Isaac Sim 运行探测常开（不依赖相机桥）
        if hasattr(self, "_sim_frame_timer") and not self._sim_frame_timer.isActive():
            self._sim_frame_timer.start()
        self._refresh_sim_frame_status()

    def _update_isaac_sim_status_ui(self) -> None:
        """刷新顶部「Isaac Sim: …」运行状态。"""
        if not hasattr(self, "sim_isaac_status_label"):
            return
        try:
            status = _eai_adapters_simulation_robodojo.RoboDojoEvalLauncher.isaac_sim_status()
        except Exception:
            self.sim_isaac_status_label.setText("Isaac Sim: 探测失败")
            self.sim_isaac_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
            return
        clients = status.get("clients") or []
        kits = status.get("kits") or []
        persist_pid = int(status.get("persist_pid") or 0)
        managed = bool(
            getattr(self, "_sim_eval_launcher", None)
            and self._sim_eval_launcher.is_running()
        )
        adopted = bool(
            getattr(self, "_sim_eval_launcher", None)
            and getattr(self._sim_eval_launcher, "_adopted", False)
        )
        if not status.get("running") and not managed:
            self.sim_isaac_status_label.setText("Isaac Sim: 未运行")
            self.sim_isaac_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
            return
        bits: _eai_platform_bootstrap.List[str] = []
        if kits:
            bits.append(f"Kit×{len(kits)}")
        if clients:
            task = clients[-1].get("task") or "?"
            bits.append(f"eval×{len(clients)}({task})")
        if persist_pid > 1:
            bits.append(f"persist={persist_pid}")
        if managed:
            bits.append("已接管" if adopted else "本窗口")
        detail = " · ".join(bits) if bits else "运行中"
        self.sim_isaac_status_label.setText(f"Isaac Sim: 运行中（{detail}）")
        self.sim_isaac_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")

    def _adopt_persisted_robodojo_eval(self) -> None:
        """界面打开时接管上次退出留下的 Isaac / 评测。"""
        if not getattr(self, "_sim_eval_launcher", None):
            return
        if self._sim_eval_launcher.is_running():
            self._update_isaac_sim_status_ui()
            return
        status = _eai_adapters_simulation_robodojo.RoboDojoEvalLauncher.isaac_sim_status()
        if not status.get("running"):
            self._update_isaac_sim_status_ui()
            return
        task = str(
            self.sim_eval_task_combo.currentData()
            or self.sim_eval_task_combo.currentText()
            or ""
        ).strip()
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                self.sim_bridge_dir_edit.text().strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
            )
        )
        adopted = self._sim_eval_launcher.adopt_running_eval(
            task,
            bridge_dir=bridge_dir,
            prefer_task=task,
            use_gui=True,
        )
        if adopted is None:
            adopted = self._sim_eval_launcher.adopt_persisted_eval(
                task,
                bridge_dir=bridge_dir,
                use_gui=True,
            )
        if adopted is not None:
            cur = str(adopted.get("task") or task or "?").strip()
            self._append_sim_log(
                f"界面启动：已接管已有 Isaac/评测（task={cur}），未重新启动"
            )
            self._activate_sim_camera_preview(force=False)
        else:
            kits = status.get("kits") or []
            clients = status.get("clients") or []
            self._append_sim_log(
                "[WARN] 检测到 Isaac/评测在跑但未能接管："
                f"Kit×{len(kits)} eval×{len(clients)}；"
                "启动评测时将拒绝重复拉起，可先「停止评测」再开。"
            )
        self._update_sim_eval_run_ui()
        self._update_isaac_sim_status_ui()

    def _refresh_sim_frame_status(self) -> None:
        self._update_isaac_sim_status_ui()
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(self.sim_bridge_dir_edit.text().strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT)
        )
        if not _eai_platform_bootstrap.os.path.isdir(bridge_dir):
            self.sim_frame_status_label.setText(f"目录不存在: {bridge_dir}")
            return
        parts: _eai_platform_bootstrap.List[str] = []
        now = _eai_platform_bootstrap.time.time()
        for key in _eai_configuration_policies.ISAAC_CAM_BRIDGE_KEYS:
            npy = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{key}.npy")
            depth_npy = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{key}_depth.npy")
            stamp = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{key}.npy.stamp")
            if not _eai_platform_bootstrap.os.path.isfile(stamp):
                stamp = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{key}.stamp")
            if not _eai_platform_bootstrap.os.path.isfile(npy):
                parts.append(f"{key}=无")
                continue
            age = "?"
            try:
                if _eai_platform_bootstrap.os.path.isfile(stamp):
                    mtime = _eai_platform_bootstrap.os.path.getmtime(stamp)
                else:
                    mtime = _eai_platform_bootstrap.os.path.getmtime(npy)
                age = f"{max(0.0, now - mtime):.1f}s"
            except OSError:
                pass
            depth_note = "+D" if _eai_platform_bootstrap.os.path.isfile(depth_npy) else ""
            parts.append(f"{key}={age}{depth_note}")
        self.sim_frame_status_label.setText(
            f"{bridge_dir}  |  " + "  ".join(parts)
            + "  → /camera/*_color + *_depth"
        )

    def _set_sim_eval_ckpt_path(self, path: str) -> None:
        """Write a full directory path into the ckpt box."""
        text = _eai_configuration_policies._ckpt_dir_abspath(path) if path else ""
        if text:
            self.sim_eval_ckpt_edit.setText(text)
        elif path:
            self.sim_eval_ckpt_edit.setText(path)

    def _sanitize_sim_eval_policy_ckpt_pair(self) -> None:
        """After UI restore: keep each policy on its own full ckpt directory."""
        policy = str(self.sim_eval_policy_combo.currentData() or "")
        ckpt = self.sim_eval_ckpt_edit.text().strip()
        if not policy:
            return
        if "starVLA" in policy:
            if (not ckpt) or (not _eai_configuration_policies._looks_like_starvla_ckpt(ckpt)) or (
                not _eai_platform_bootstrap.os.path.isabs(ckpt)
            ):
                self._set_sim_eval_ckpt_path(_eai_configuration_policies._default_starvla_ckpt_text())
            else:
                self._set_sim_eval_ckpt_path(ckpt)
            act_idx = self.sim_eval_action_combo.findData("joint")
            if act_idx >= 0:
                self.sim_eval_action_combo.setCurrentIndex(act_idx)
            return
        if _eai_configuration_policies._is_pi05_robodojo_policy(policy) and (
            (not ckpt)
            or ckpt == "demo"
            or (not _eai_platform_bootstrap.os.path.isabs(ckpt))
            or "RoboDojo-cotrain" in ckpt
            or _eai_configuration_policies._ckpt_owned_by_lingbot_va_not_pi05(ckpt)
            or _eai_configuration_policies._looks_like_starvla_ckpt(ckpt)
        ):
            self._set_sim_eval_ckpt_path(_eai_configuration_policies._default_pi05_ckpt_text())
            act_idx = self.sim_eval_action_combo.findData("joint")
            if act_idx >= 0:
                self.sim_eval_action_combo.setCurrentIndex(act_idx)
            return
        if _eai_configuration_models.is_lingbot_va_policy(policy) and (
            (not ckpt) or (not _eai_platform_bootstrap.os.path.isabs(ckpt)) or ckpt in ("demo", "hf_qwenpi_v3")
        ):
            self._set_sim_eval_ckpt_path(_eai_configuration_policies._default_lingbot_va_ckpt_text())
            return
        if (not _eai_configuration_models.is_lingbot_va_policy(policy)) and _eai_configuration_policies._ckpt_owned_by_lingbot_va_not_pi05(
            ckpt
        ):
            self._on_sim_eval_policy_changed()

    def _update_sim_eval_run_ui(self, *_args) -> None:
        running = self._sim_eval_launcher.is_running()
        was_running = bool(getattr(self, "_sim_eval_was_running", False))
        stopping = bool(getattr(self._sim_eval_launcher, "_stopping", False))
        no_policy = not str(self.sim_eval_policy_combo.currentData() or "")
        # 「停止评测」在探测到系统里仍有 Isaac/eval 时也应可点（含外部残留）
        try:
            ext_running = bool(_eai_adapters_simulation_robodojo.RoboDojoEvalLauncher.isaac_sim_status().get("running"))
        except Exception:
            ext_running = False
        self.sim_eval_stop_run_btn.setEnabled(running or ext_running or stopping)
        self._update_isaac_sim_status_ui()
        self._update_sim_pause_ui()
        if hasattr(self, "sim_eval_viewer_only_btn"):
            self.sim_eval_viewer_only_btn.setEnabled(
                (not running)
                and (not stopping)
                and self._sim_backend_id() == "isaac"
            )
        # 策略 / GUI 模式仍锁定；任务可在运行中切换
        self.sim_eval_policy_combo.setEnabled(not running)
        self.sim_eval_ckpt_edit.setEnabled(not running and not no_policy)
        self.sim_eval_action_combo.setEnabled(not running and not no_policy)
        if hasattr(self, "sim_eval_use_gui_check"):
            self.sim_eval_use_gui_check.setEnabled(not running)
        if hasattr(self, "sim_eval_bundled_check"):
            self.sim_eval_bundled_check.setEnabled(not running and not no_policy)
        if stopping:
            self.sim_eval_run_status.setText("评测: 正在停止…")
            self.sim_eval_run_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
            self.sim_eval_start_btn.setText("启动评测")
            self.sim_eval_start_btn.setEnabled(False)
            self.sim_eval_task_combo.setEnabled(False)
        elif running:
            cur = self._sim_eval_launcher.current_task() or "?"
            adopted = bool(getattr(self._sim_eval_launcher, "_adopted", False))
            mode = "复用外部进程" if adopted else "常驻运行中"
            self.sim_eval_run_status.setText(
                f"评测: {mode}（当前任务 {cur}；可「切换任务」）"
            )
            self.sim_eval_run_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self.sim_eval_start_btn.setText("切换任务")
            self.sim_eval_start_btn.setEnabled(True)
            self.sim_eval_task_combo.setEnabled(True)
            if not self._sim_npy_preview_timer.isActive():
                self._sim_npy_preview_timer.start()
            self._update_enable_status_ui()
            self._update_arm_move_btns_ui()
        else:
            self.sim_eval_run_status.setText("评测: 空闲")
            self.sim_eval_run_status.setStyleSheet("")
            self.sim_eval_start_btn.setText("启动评测")
            self.sim_eval_start_btn.setEnabled(True)
            self.sim_eval_task_combo.setEnabled(True)
            self._cancel_sim_arm_move("")
            if _eai_configuration_policies.clear_gui_robot_cmd is not None:
                try:
                    _eai_configuration_policies.clear_gui_robot_cmd(self._sim_bridge_dir())
                except Exception:
                    pass
            self._update_enable_status_ui()
            self._update_arm_move_btns_ui()
            # 评测刚结束：刷新结果树，提示可合并进榜单对比
            if was_running and not stopping:
                try:
                    self._refresh_sim_eval_tree()
                except Exception:
                    pass
                if hasattr(self, "status_bar") and self.status_bar is not None:
                    self.status_bar.showMessage(
                        "评测已结束：在结果中选中含 _result.json 的运行，点「合并到榜单」"
                        "可与业内/实时榜并排对比",
                        8000,
                    )
            # 评测未跑时：其它后端/相机桥/残留共享帧仍可继续预览
            if self._any_sim_process_running() or (
                getattr(self, "_sim_bridge_launcher", None) is not None
                and self._sim_bridge_launcher.is_running()
            ):
                if not self._sim_npy_preview_timer.isActive():
                    self._sim_npy_preview_timer.start()
            else:
                age = self._sim_bridge_newest_age_s()
                if age is not None and age <= 900.0:
                    if not self._sim_npy_preview_timer.isActive():
                        self._sim_npy_preview_timer.start()
                    if self._sim_backend_id() == "isaac":
                        self.sim_eval_run_status.setText(
                            f"评测: 空闲（共享帧约 {age:.0f}s 前，预览仍可读）"
                        )
                        self.sim_eval_run_status.setStyleSheet(
                            f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};"
                        )
                else:
                    self._sim_npy_preview_timer.stop()
                    self._sim_preview_retry_timer.stop()
                    self._sim_preview_retry_left = 0
        self._sim_eval_was_running = bool(running)

    def _ensure_workspace_expanded(self) -> None:
        if getattr(self, "_workspace_collapsed", False):
            self._toggle_workspace_panel()

    def _reset_sim_bridge_frames(self) -> None:
        """清掉旧共享帧，避免预览一直显示上次评测的静止画面。"""
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                self.sim_bridge_dir_edit.text().strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
            )
        )
        _eai_platform_bootstrap.os.makedirs(bridge_dir, exist_ok=True)
        removed = 0
        for key in _eai_configuration_policies.ISAAC_CAM_BRIDGE_KEYS:
            for base in (key, f"{key}_depth"):
                for suffix in (".npy", ".npy.stamp", ".npy.writing", ".stamp"):
                    path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{base}{suffix}")
                    try:
                        if _eai_platform_bootstrap.os.path.isfile(path):
                            _eai_platform_bootstrap.os.remove(path)
                            removed += 1
                    except OSError:
                        pass
        self._sim_npy_last_mtime.clear()
        for topic in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS:
            self._frame_cache.pop(topic, None)
            panel = self.panels.get(topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                try:
                    panel.update_frame(_eai_platform_bootstrap.np.zeros((240, 320, 3), dtype=_eai_platform_bootstrap.np.uint8))
                except Exception:
                    pass
            elif isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D):
                try:
                    panel.update_depth(_eai_platform_bootstrap.np.zeros((240, 320), dtype=_eai_platform_bootstrap.np.uint16))
                except Exception:
                    pass
        self._append_sim_log(
            f"已清空共享帧目录 {bridge_dir}（删除 {removed} 个旧文件），等待本轮 Isaac 写出"
        )

    def _purge_sim_bridge_cam_keys(self, cam_keys: _eai_platform_bootstrap.Sequence[str]) -> int:
        """删除指定相机的共享彩色/深度帧，避免无摄像头仍显示旧图。"""
        bridge_dir = self._sim_bridge_dir_resolved()
        if not _eai_platform_bootstrap.os.path.isdir(bridge_dir):
            return 0
        removed = 0
        for cam_key in cam_keys:
            for name in (f"{cam_key}.npy", f"{cam_key}_depth.npy"):
                for path in (
                    _eai_platform_bootstrap.os.path.join(bridge_dir, name),
                    _eai_platform_bootstrap.os.path.join(bridge_dir, name + ".stamp"),
                ):
                    try:
                        if _eai_platform_bootstrap.os.path.isfile(path):
                            _eai_platform_bootstrap.os.remove(path)
                            removed += 1
                    except OSError:
                        pass
            for _ck, topic in _eai_configuration_policies.SIM_PREVIEW_CAM_TOPICS + _eai_configuration_policies.SIM_PREVIEW_DEPTH_CAM_TOPICS:
                if _ck != cam_key:
                    continue
                self._frame_cache.pop(topic, None)
                self._sim_npy_last_mtime.pop(topic, None)
        return removed

    def _sync_sim_preview_topics_to_bridge(self, *, max_age_s: float = 120.0) -> bool:
        """只勾选/显示共享目录里实际有画面的头/左腕/右腕；无摄像头则不显示。

        返回是否改动了勾选（从而需要 rebuild）。
        """
        if any(t not in self.topic_checks for t in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS):
            self._on_topics_updated(dict(getattr(self, "_topic_types", {}) or {}))
        owned = bool(self._any_sim_process_running())
        age = float(max_age_s)
        if not owned:
            age = max(age, 900.0)
        color_keys = self._sim_bridge_fresh_cam_keys(max_age_s=age)
        depth_keys = self._sim_bridge_fresh_depth_keys(max_age_s=age)
        want: set = set()
        for cam_key, topic in _eai_configuration_policies.SIM_PREVIEW_CAM_TOPICS:
            if cam_key in color_keys:
                want.add(topic)
        for cam_key, topic in _eai_configuration_policies.SIM_PREVIEW_DEPTH_CAM_TOPICS:
            if cam_key in depth_keys:
                want.add(topic)

        prev = getattr(self, "_sim_preview_want_topics", None)
        want_frozen = frozenset(want)
        if prev == want_frozen:
            return False
        self._sim_preview_want_topics = want_frozen

        changed = False
        for topic in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS:
            checkbox = self.topic_checks.get(topic)
            if checkbox is None:
                continue
            should = topic in want
            if should and topic in self._topic_user_unchecked:
                # 用户手动取消过：有帧也不强行打开
                should = False
            if not should and topic not in want:
                # 无摄像头：清掉「用户取消」标记，以便下次有帧可自动出现
                self._topic_user_unchecked.discard(topic)
            if checkbox.isChecked() != should:
                checkbox.blockSignals(True)
                checkbox.setChecked(should)
                checkbox.blockSignals(False)
                changed = True
        if changed:
            enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
            self._apply_selection(enabled)
            self._sync_mv6d_preview_visibility()
        return changed

    def _sync_mv6d_preview_visibility(self) -> None:
        """多视角 6D：始终保留头/左腕/右腕预览格；无帧时显示占位文案（不隐藏）。"""
        mapping = (
            (
                "/camera/head_color",
                "mv6d_head_preview",
                "_mv6d_head_pix",
                "头部 · 等待画面\n勾选 /camera/head_color 或开仿真预览",
            ),
            (
                "left",
                "mv6d_left_wrist_preview",
                "_mv6d_left_wrist_pix",
                "左腕 · 等待画面",
            ),
            (
                "right",
                "mv6d_right_wrist_preview",
                "_mv6d_right_wrist_pix",
                "右腕 · 等待画面",
            ),
        )
        for topic, attr, cache_attr, placeholder in mapping:
            lab = getattr(self, attr, None)
            if lab is None:
                continue
            lab.setVisible(True)
            if self._pick_mv6d_color_frame(topic) is not None:
                continue
            # 无帧：清掉旧 pixmap，露出占位说明，避免空白黑块
            if getattr(self, cache_attr, None) is None:
                self._set_mv6d_preview_label(
                    lab, None, placeholder, cache_attr=cache_attr
                )

    def _activate_sim_camera_preview(self, force: bool = True) -> None:
        """展开图像预览；仅显示共享目录里实际有帧的头/腕相机（无则不占位）。

        所有仿真后端共用 ISAAC_CAM_BRIDGE_DIR；写出 cam_*.npy 的后端（Isaac /
        LIBERO 等）都会刷新 /camera/*，从而驱动多视角 6D 预览。
        """
        self._ensure_workspace_expanded()
        if force:
            for topic in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS:
                self._topic_user_unchecked.discard(topic)
            self._sim_preview_want_topics = None
        # ROS 可能一直发现不到 /camera：先把仿真伪 topic 注入左侧列表
        if any(t not in self.topic_checks for t in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS):
            self._on_topics_updated(dict(getattr(self, "_topic_types", {}) or {}))
        # 先按现有帧勾选；尚无帧时先全不勾，等 poll 到 npy 再出现
        if not self._sync_sim_preview_topics_to_bridge(
            max_age_s=900.0 if not force else 120.0
        ):
            enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
            # 去掉无帧的仿真相机占位
            for topic in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS:
                if topic not in (getattr(self, "_sim_preview_want_topics", None) or set()):
                    enabled.discard(topic)
                    cb = self.topic_checks.get(topic)
                    if cb is not None and cb.isChecked():
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
            self._apply_selection(enabled)
            self._sync_mv6d_preview_visibility()
        if not self._sim_npy_preview_timer.isActive():
            self._sim_npy_preview_timer.start()
        self._sim_preview_retry_left = 20
        if not self._sim_preview_retry_timer.isActive():
            self._sim_preview_retry_timer.start()
        if getattr(self, "status_bar", None) is not None:
            keys = sorted(self._sim_bridge_fresh_cam_keys(max_age_s=900.0))
            label = "、".join(keys) if keys else "等待 cam_*.npy"
            self.status_bar.showMessage(
                f"已打开仿真预览（仅显示有画面的相机：{label}）",
                5000,
            )

    def _maybe_resume_sim_npy_preview(self) -> None:
        """UI 重启后：若共享帧仍在，自动恢复预览（不依赖本窗口是否拥有评测进程）。"""
        age = self._sim_bridge_newest_age_s()
        if age is None:
            return
        if age > 600.0 and not self._any_sim_process_running():
            return
        self._activate_sim_camera_preview(force=False)
        self._append_sim_log(
            f"检测到共享帧目录（最新约 {age:.0f}s 前），已恢复图像预览轮询"
            + ("；帧已较旧，若画面静止请在本窗口重新启动评测" if age > 5.0 else "")
        )

    def _retry_enable_sim_preview_topics(self) -> None:
        """topic 列表稍后才发现时，按共享目录实际有帧的相机补勾（无摄像头不显示）。"""
        if self._sim_preview_retry_left <= 0 or not (
            self._any_sim_process_running()
            or self._sim_bridge_newest_age_s() is not None
        ):
            self._sim_preview_retry_timer.stop()
            self._sim_preview_retry_left = 0
            return
        self._sim_preview_retry_left -= 1
        if any(t not in self.topic_checks for t in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS):
            self._on_topics_updated(dict(getattr(self, "_topic_types", {}) or {}))
        self._sim_preview_want_topics = None  # 允许按新帧重算
        self._sync_sim_preview_topics_to_bridge()
        found = sum(1 for t in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS if t in self.topic_checks)
        if found >= len(_eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS) and self._sim_bridge_fresh_cam_keys(
            max_age_s=900.0
        ):
            self._sim_preview_retry_timer.stop()
            self._sim_preview_retry_left = 0

    def _poll_sim_npy_preview(self) -> None:
        """直读 ISAAC_CAM_BRIDGE_DIR 下 RGB / depth .npy，刷新图像预览（不依赖 ROS）。

        对所有仿真后端生效：只要共享目录有新帧就更新 /camera/*（含多视角 6D）。
        """
        owned = bool(self._any_sim_process_running()) or (
            getattr(self, "_sim_bridge_launcher", None) is not None
            and self._sim_bridge_launcher.is_running()
        )
        bridge_dir = _eai_platform_bootstrap.os.path.abspath(
            _eai_platform_bootstrap.os.path.expanduser(
                self.sim_bridge_dir_edit.text().strip() or _eai_configuration_policies.ISAAC_CAM_BRIDGE_DIR_DEFAULT
            )
        )
        if not _eai_platform_bootstrap.os.path.isdir(bridge_dir):
            return
        # UI 崩溃重启后评测可能仍是外部孤儿进程：只要目录里有帧就继续读
        if not owned and self._sim_bridge_newest_age_s() is None:
            return
        # 保证左侧 Topic 列表存在（不依赖 ROS discovery）；不强制改用户勾选
        if any(t not in self.topic_checks for t in _eai_configuration_policies.SIM_PREVIEW_ALL_TOPICS):
            self._on_topics_updated(dict(getattr(self, "_topic_types", {}) or {}))
        enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
        missing = [t for t in enabled if t not in self.panels]
        if missing:
            self._apply_selection(enabled)

        now = _eai_platform_bootstrap.time.time()
        # 本窗口拥有评测时丢掉过旧帧；外部残留帧放宽到 15min 以便重启后仍能看到最后一帧
        max_age = 120.0 if owned else 900.0
        fresh = 0
        updated_color_topics: _eai_platform_bootstrap.List[str] = []
        for cam_key, topic in _eai_configuration_policies.SIM_PREVIEW_CAM_TOPICS:
            npy_path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{cam_key}.npy")
            stamp_path = npy_path + ".stamp"
            try:
                mtime = _eai_platform_bootstrap.os.path.getmtime(
                    stamp_path if _eai_platform_bootstrap.os.path.isfile(stamp_path) else npy_path
                )
            except OSError:
                # 无文件：清掉可能残留的缓存，避免无摄像头仍被 6D 融合
                self._frame_cache.pop(topic, None)
                self._sim_npy_last_mtime.pop(topic, None)
                continue
            if now - mtime > max_age:
                self._frame_cache.pop(topic, None)
                self._sim_npy_last_mtime.pop(topic, None)
                continue
            if self._sim_npy_last_mtime.get(topic) == mtime:
                fresh += 1
                continue
            try:
                with open(npy_path, "rb") as f:
                    arr = _eai_platform_bootstrap.np.load(f)
                arr = _eai_platform_bootstrap.np.asarray(arr)
            except Exception:
                continue
            if arr.ndim != 3 or arr.shape[2] < 3:
                continue
            rgb = _eai_platform_bootstrap.np.ascontiguousarray(arr[..., :3], dtype=_eai_platform_bootstrap.np.uint8)
            bgr = _eai_platform_bootstrap.cv2.cvtColor(rgb, _eai_platform_bootstrap.cv2.COLOR_RGB2BGR)
            self._sim_npy_last_mtime[topic] = mtime
            self._frame_cache[topic] = bgr
            self._received_topics[topic] = self._received_topics.get(topic, 0) + 1
            panel = self.panels.get(topic)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                panel.update_frame(bgr)
            updated_color_topics.append(topic)
            fresh += 1

        for cam_key, topic in _eai_configuration_policies.SIM_PREVIEW_DEPTH_CAM_TOPICS:
            npy_path = _eai_platform_bootstrap.os.path.join(bridge_dir, f"{cam_key}_depth.npy")
            stamp_path = npy_path + ".stamp"
            try:
                mtime = _eai_platform_bootstrap.os.path.getmtime(
                    stamp_path if _eai_platform_bootstrap.os.path.isfile(stamp_path) else npy_path
                )
            except OSError:
                continue
            if now - mtime > max_age:
                continue
            # 丢弃明显旧于同相机彩色的 depth（LIBERO 只写 RGB 时的 Isaac 残留）
            color_topic = None
            for ck, ct in _eai_configuration_policies.SIM_PREVIEW_CAM_TOPICS:
                if ck == cam_key:
                    color_topic = ct
                    break
            if color_topic is not None:
                color_mtime = self._sim_npy_last_mtime.get(color_topic)
                if color_mtime is not None and (float(color_mtime) - float(mtime)) > 2.0:
                    # 清掉缓存里可能残留的旧 depth
                    self._frame_cache.pop(topic, None)
                    continue
            if self._sim_npy_last_mtime.get(topic) == mtime:
                fresh += 1
                continue
            try:
                with open(npy_path, "rb") as f:
                    arr = _eai_platform_bootstrap.np.load(f)
                arr = _eai_platform_bootstrap.np.asarray(arr)
            except Exception:
                continue
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.ndim != 2:
                continue
            if color_topic is not None:
                color = self._frame_cache.get(color_topic)
                if color is not None and not self._depth_compatible_with_color(color, arr):
                    self._frame_cache.pop(topic, None)
                    continue
            if arr.dtype != _eai_platform_bootstrap.np.uint16:
                depth_f = arr.astype(_eai_platform_bootstrap.np.float32)
                finite = depth_f[_eai_platform_bootstrap.np.isfinite(depth_f)]
                if finite.size and float(_eai_platform_bootstrap.np.nanmax(finite)) < 100.0:
                    arr = _eai_platform_bootstrap.np.clip(depth_f * 1000.0, 0, 65535).astype(_eai_platform_bootstrap.np.uint16)
                else:
                    arr = _eai_platform_bootstrap.np.clip(depth_f, 0, 65535).astype(_eai_platform_bootstrap.np.uint16)
            depth = _eai_platform_bootstrap.np.ascontiguousarray(arr)
            self._sim_npy_last_mtime[topic] = mtime
            self._frame_cache[topic] = depth
            self._received_topics[topic] = self._received_topics.get(topic, 0) + 1
            panel = self.panels.get(topic)
            if isinstance(panel, _eai_ui_cameras_panels.DepthPanel3D):
                panel.update_depth(depth)
            fresh += 1

        if updated_color_topics:
            self._maybe_refresh_mv6d_previews_for_topics(updated_color_topics)

        # 有新帧或帧消失时：只显示实际有摄像头画面的头/腕
        self._sync_sim_preview_topics_to_bridge(
            max_age_s=120.0 if owned else 900.0
        )

        if fresh == 0 and getattr(self, "_sim_preview_wait_logged", False) is False:
            self._sim_preview_wait_logged = True
            self._append_sim_log(
                "图像预览：尚无本轮共享帧（需评测写出 ISAAC_CAM_BRIDGE_DIR/*.npy）"
            )

