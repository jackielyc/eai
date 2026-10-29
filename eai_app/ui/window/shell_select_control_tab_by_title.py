"""ui.window.shell select control tab by title — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_chat = module_ref('eai_app.configuration.chat')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_cameras_panels = module_ref('eai_app.ui.cameras.panels')

class ShellSelectControlTabByTitleMixin:
    def _select_control_tab_by_title(self, title: str) -> bool:
        title = (title or "").strip()
        if not title:
            return False
        tabs = getattr(self, "control_tabs", None)
        if tabs is None:
            return False
        outer_idx = self._index_of_control_tab(title)
        if outer_idx >= 0:
            tabs.blockSignals(True)
            try:
                tabs.setCurrentIndex(outer_idx)
            finally:
                tabs.blockSignals(False)
            return True
        misc_idx = self._index_of_misc_child(title)
        if misc_idx < 0:
            return False
        other_idx = self._index_of_control_tab(_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE)
        if other_idx < 0:
            return False
        tabs.blockSignals(True)
        try:
            tabs.setCurrentIndex(other_idx)
        finally:
            tabs.blockSignals(False)
        inner = self.misc_inner_tabs
        inner.blockSignals(True)
        try:
            inner.setCurrentIndex(misc_idx)
        finally:
            inner.blockSignals(False)
        return True

    def _persist_control_tabs_layout(self) -> None:
        if getattr(self, "_control_tabs_order_guard", False):
            return
        titles = self._control_tab_titles()
        if not titles:
            return
        try:
            prev = _eai_infrastructure_ui_storage.load_control_tabs_order()
            merged: _eai_platform_bootstrap.List[str] = list(titles)
            seen = set(merged)
            for title in prev:
                if title not in seen:
                    merged.append(title)
                    seen.add(title)
            for title in _eai_configuration_tabs.CONTROL_TAB_TITLES:
                if title not in seen:
                    merged.append(title)
                    seen.add(title)
            misc_children = self._misc_child_titles()
            path = _eai_infrastructure_ui_storage.save_control_tabs_order(merged, misc_children=misc_children)
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"已保存 tab 布局 → {path}", 3000)
        except Exception as exc:
            status_bar = getattr(self, "status_bar", None)
            if status_bar is not None:
                status_bar.showMessage(f"保存 tab 布局失败: {exc}", 4000)

    def _nest_control_tab_into_misc(self, title: str) -> None:
        title = (title or "").strip()
        if not title or title == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE or title == "…":
            return
        if self._index_of_misc_child(title) >= 0:
            other_idx = self._index_of_control_tab(_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE)
            if other_idx >= 0:
                self.control_tabs.setCurrentIndex(other_idx)
                self.misc_inner_tabs.setCurrentIndex(self._index_of_misc_child(title))
            return
        tabs = getattr(self, "control_tabs", None)
        inner = getattr(self, "misc_inner_tabs", None)
        if tabs is None or inner is None:
            return
        idx = self._index_of_control_tab(title)
        if idx < 0:
            return
        widget = tabs.widget(idx)
        if widget is None:
            return
        tip = tabs.tabToolTip(idx)
        self._control_tabs_order_guard = True
        try:
            tabs.removeTab(idx)
            new_i = inner.addTab(widget, title)
            if tip:
                inner.setTabToolTip(new_i, tip)
            other_idx = self._index_of_control_tab(_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE)
            if other_idx >= 0:
                tabs.setCurrentIndex(other_idx)
            inner.setCurrentIndex(new_i)
        finally:
            self._control_tabs_order_guard = False
        self._update_misc_empty_hint()
        self._refresh_control_tab_tooltips()
        self._persist_control_tabs_layout()
        status_bar = getattr(self, "status_bar", None)
        if status_bar is not None:
            status_bar.showMessage(
                f"已将「{title}」移入「{_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE}」", 2500
            )

    def _pop_control_tab_from_misc(self, title: str, insert_at: int = -1) -> None:
        title = (title or "").strip()
        if not title or title == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE:
            return
        tabs = getattr(self, "control_tabs", None)
        inner = getattr(self, "misc_inner_tabs", None)
        if tabs is None or inner is None:
            return
        if self._index_of_control_tab(title) >= 0:
            return
        misc_idx = self._index_of_misc_child(title)
        if misc_idx < 0:
            return
        widget = inner.widget(misc_idx)
        if widget is None:
            return
        tip = inner.tabToolTip(misc_idx)
        self._control_tabs_order_guard = True
        try:
            inner.removeTab(misc_idx)
            other_idx = self._index_of_control_tab(_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE)
            if insert_at < 0:
                target = other_idx if other_idx >= 0 else tabs.count()
            else:
                target = max(0, min(int(insert_at), tabs.count()))
            new_i = tabs.insertTab(target, widget, title)
            if tip:
                tabs.setTabToolTip(new_i, tip)
            tabs.setCurrentIndex(new_i)
        finally:
            self._control_tabs_order_guard = False
        self._update_misc_empty_hint()
        self._refresh_control_tab_tooltips()
        self._persist_control_tabs_layout()
        status_bar = getattr(self, "status_bar", None)
        if status_bar is not None:
            status_bar.showMessage(f"已将「{title}」移回顶栏", 2500)

    def _restore_control_misc_children(self) -> None:
        """按 eai/control_tabs_order.json 的 misc_children 收入「其他」。"""
        if getattr(self, "_only_tabs", None):
            return
        children = _eai_infrastructure_ui_storage.load_control_misc_children()
        if not children:
            self._update_misc_empty_hint()
            return
        self._control_tabs_order_guard = True
        try:
            for title in children:
                # 内联移动，避免中途反复写盘
                if not title or title == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE:
                    continue
                if self._index_of_misc_child(title) >= 0:
                    continue
                tabs = self.control_tabs
                inner = self.misc_inner_tabs
                idx = self._index_of_control_tab(title)
                if idx < 0:
                    continue
                widget = tabs.widget(idx)
                if widget is None:
                    continue
                tip = tabs.tabToolTip(idx)
                tabs.removeTab(idx)
                new_i = inner.addTab(widget, title)
                if tip:
                    inner.setTabToolTip(new_i, tip)
        finally:
            self._control_tabs_order_guard = False
        self._update_misc_empty_hint()

    def _refresh_control_tab_tooltips(self) -> None:
        """为每个控制区 tab 设置悬停提示：可用 --tab 如何启动。"""
        tabs = getattr(self, "control_tabs", None)
        if tabs is None:
            return
        for i in range(tabs.count()):
            title = tabs.tabText(i)
            tabs.setTabToolTip(i, _eai_configuration_tabs.format_control_tab_launch_tooltip(title))
        inner = getattr(self, "misc_inner_tabs", None)
        if inner is None:
            return
        for i in range(inner.count()):
            title = inner.tabText(i)
            tip = _eai_configuration_tabs.format_control_tab_launch_tooltip(title)
            tip += f"\n\n当前位于「{_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE}」；拖回顶栏或右键 → 移回顶栏"
            inner.setTabToolTip(i, tip)

    def _restore_control_tabs_order(self) -> None:
        """按 eai/control_tabs_order.json 恢复当前可见 tab 顺序。"""
        tabs = getattr(self, "control_tabs", None)
        if tabs is None or tabs.count() <= 1:
            return
        desired = _eai_infrastructure_ui_storage.preferred_control_tabs_order(self._control_tab_titles())
        if not desired or desired == self._control_tab_titles():
            return
        self._control_tabs_order_guard = True
        try:
            for target_idx, title in enumerate(desired):
                cur = -1
                for i in range(tabs.count()):
                    if tabs.tabText(i) == title:
                        cur = i
                        break
                if cur >= 0 and cur != target_idx:
                    tabs.tabBar().moveTab(cur, target_idx)
        finally:
            self._control_tabs_order_guard = False

    def _save_viewer_ui_state(self) -> str:
        """关闭前把所有可操作控件状态写入 eai/viewer_ui_state.json。"""
        if getattr(self, "_viewer_ui_state_saved", False):
            return _eai_configuration_chat.VIEWER_UI_STATE_PATH
        if not getattr(self, "_heavy_ui_built", False):
            return _eai_configuration_chat.VIEWER_UI_STATE_PATH
        widgets: _eai_platform_bootstrap.Dict[str, object] = {}
        for prefix, owner in self._viewer_ui_state_owners():
            widgets.update(_eai_infrastructure_ui_storage._ui_state_collect_owner(owner, prefix=prefix))

        geo = self.geometry()
        control_tab = self._current_control_tab_title()

        splitter = getattr(self, "_main_splitter", None)
        splitter_sizes: _eai_platform_bootstrap.List[int] = []
        if splitter is not None:
            try:
                splitter_sizes = [int(x) for x in splitter.sizes()]
            except Exception:
                splitter_sizes = []

        topic_checks = {
            str(topic): bool(cb.isChecked())
            for topic, cb in getattr(self, "topic_checks", {}).items()
        }
        topic_unchecked = sorted(
            str(t) for t in getattr(self, "_topic_user_unchecked", set())
        )

        payload: _eai_platform_bootstrap.Dict[str, object] = {
            "version": 1,
            "arm_move_speed_mapping_version": 2,
            "widgets": widgets,
            "control_tab": control_tab,
            "splitter_sizes": splitter_sizes,
            "geometry": [geo.x(), geo.y(), geo.width(), geo.height()],
            "topic_checks": topic_checks,
            "topic_user_unchecked": topic_unchecked,
            "workspace_collapsed": bool(
                getattr(self, "_workspace_collapsed", True)
            ),
        }
        path = _eai_infrastructure_ui_storage.save_viewer_ui_state(payload)
        self._viewer_ui_state_saved = True
        status_bar = getattr(self, "status_bar", None)
        if status_bar is not None:
            status_bar.showMessage(
                f"已保存界面状态（{len(widgets)} 项）→ {path}", 3000
            )
        return path

    def _restore_viewer_ui_state(self) -> None:
        """启动时恢复上次关闭前的控件状态。"""
        state = _eai_infrastructure_ui_storage.load_viewer_ui_state()
        if not state:
            return

        widgets = state.get("widgets")
        if isinstance(widgets, dict):
            widgets = dict(widgets)
            widgets.pop("mv6d_gt_enable_check", None)
            saved_speed = widgets.get('arm_move_speed_slider')
            if state.get('arm_move_speed_mapping_version', 1) < 2 and isinstance(saved_speed, dict):
                old_value = max(10, min(100, int(saved_speed.get('value', 46))))
                old_speed = .1 + (old_value/100.)*.9
                widgets['arm_move_speed_slider'] = dict(saved_speed, value=_eai_adapters_robot_geometry.arm_move_speed_to_slider(old_speed))
        restored = 0
        pending_combo: _eai_platform_bootstrap.Dict[str, object] = {}
        if isinstance(widgets, dict):
            for prefix, owner in self._viewer_ui_state_owners():
                restored += _eai_infrastructure_ui_storage._ui_state_apply_owner(owner, widgets, prefix=prefix)
            # 动态下拉（topic 等）可能尚未填满：记下 combo 以便后续重试
            for key, payload in widgets.items():
                if not isinstance(payload, dict) or payload.get("kind") != "combo":
                    continue
                applied = False
                for prefix, owner in self._viewer_ui_state_owners():
                    if prefix:
                        if not key.startswith(prefix):
                            continue
                        name = key[len(prefix) :]
                    else:
                        if "." in key or key.startswith("tab:"):
                            continue
                        name = key
                    if not name or "." in name:
                        continue
                    w = getattr(owner, name, None)
                    if isinstance(w, _eai_platform_bootstrap.QComboBox):
                        cur = w.currentText()
                        want = str(payload.get("text") or "")
                        data = payload.get("data", None)
                        if data is not None and w.currentData() == data:
                            applied = True
                        elif want and cur == want:
                            applied = True
                        break
                if not applied:
                    pending_combo[key] = payload
        self._pending_combo_state = pending_combo
        # 相对偏移旋钮已从 m 改为 cm：旧存档里 |v|≤0.5 的按米换算
        for name in ("offset_x_spin", "offset_y_spin", "offset_z_spin"):
            spin = getattr(self, name, None)
            if not isinstance(spin, _eai_platform_bootstrap.QDoubleSpinBox):
                continue
            v = float(spin.value())
            if 0.0 < abs(v) <= _eai_configuration_chat.MANUAL_OFFSET_MAX_M:
                spin.blockSignals(True)
                spin.setValue(v * 100.0)
                spin.blockSignals(False)

        geo = state.get("geometry")
        if isinstance(geo, list) and len(geo) == 4:
            try:
                x, y, w, h = (int(geo[0]), int(geo[1]), int(geo[2]), int(geo[3]))
                if w >= 640 and h >= 480:
                    self.setGeometry(x, y, w, h)
            except Exception:
                pass

        control_tab = str(state.get("control_tab") or "").strip()
        if control_tab:
            self._select_control_tab_by_title(control_tab)

        splitter = getattr(self, "_main_splitter", None)
        sizes = state.get("splitter_sizes")
        if splitter is not None and isinstance(sizes, list) and sizes:
            try:
                ints = [int(x) for x in sizes]
                if len(ints) == splitter.count() and sum(ints) > 0:
                    splitter.setSizes(ints)
            except Exception:
                pass

        want_collapsed = state.get("workspace_collapsed")
        if want_collapsed is not None:
            cur = bool(getattr(self, "_workspace_collapsed", True))
            if bool(want_collapsed) != cur:
                self._toggle_workspace_panel()

        topic_checks = state.get("topic_checks")
        if isinstance(topic_checks, dict):
            # 深度/腕部太多路会卡死：恢复时默认关掉 depth，彩色也只保留常用几路
            pending: _eai_platform_bootstrap.Dict[str, bool] = {}
            for k, v in topic_checks.items():
                topic = str(k)
                want = bool(v)
                if _eai_perception_image_geometry.is_depth_topic(topic):
                    want = False
                elif want and topic not in _eai_configuration_policies.UI_DEFAULT_COLOR_TOPIC_SUFFIXES:
                    # 腕部等非默认彩色：恢复时不自动开，避免 10 路齐开
                    if "wrist" in topic.lower():
                        want = False
                pending[topic] = want
            self._pending_topic_checks = pending
        unchecked = state.get("topic_user_unchecked")
        if isinstance(unchecked, list):
            self._topic_user_unchecked = {str(t) for t in unchecked if str(t).strip()}

        status_bar = getattr(self, "status_bar", None)
        if status_bar is not None and restored:
            status_bar.showMessage(f"已恢复界面状态（{restored} 项）", 3000)

    def _retry_pending_combo_ui_state(self) -> None:
        """topic 列表刷新后，重试尚未匹配的下拉框恢复。"""
        pending = getattr(self, "_pending_combo_state", None)
        if not isinstance(pending, dict) or not pending:
            return
        still: _eai_platform_bootstrap.Dict[str, object] = {}
        for key, payload in pending.items():
            ok = False
            for prefix, owner in self._viewer_ui_state_owners():
                if prefix:
                    if not key.startswith(prefix):
                        continue
                    name = key[len(prefix) :]
                else:
                    if "." in key or key.startswith("tab:"):
                        continue
                    name = key
                if not name or "." in name:
                    continue
                w = getattr(owner, name, None)
                if w is not None and _eai_infrastructure_ui_storage._ui_state_write_widget(w, payload):
                    ok = True
                    break
            if not ok:
                still[key] = payload
        self._pending_combo_state = still

    def _restore_sim_backends_order(self) -> None:
        """按 eai/sim_backends_order.json 恢复仿真后端选择条顺序。"""
        bar = getattr(self, "sim_backend_tab_bar", None)
        if bar is None or bar.count() <= 1:
            return
        desired = _eai_infrastructure_ui_storage.preferred_sim_backends_order(self._sim_backend_selector_keys())
        if not desired or desired == self._sim_backend_selector_keys():
            return
        self._sim_backend_tabs_order_guard = True
        try:
            for target_idx, key in enumerate(desired):
                cur = -1
                for i in range(bar.count()):
                    if str(bar.tabData(i) or "") == key:
                        cur = i
                        break
                if cur >= 0 and cur != target_idx:
                    bar.moveTab(cur, target_idx)
        finally:
            self._sim_backend_tabs_order_guard = False

    def _sync_sim_backend_tab_from_radios(self) -> None:
        """把隐藏 radio 的选中态同步到后端选择 TabBar。"""
        bar = getattr(self, "sim_backend_tab_bar", None)
        if bar is None or bar.count() <= 0:
            return
        key = self._sim_backend_selector_key()
        target = -1
        for i in range(bar.count()):
            if str(bar.tabData(i) or "") == key:
                target = i
                break
        if target < 0 or target == bar.currentIndex():
            return
        self._sim_backend_tabs_guard = True
        try:
            bar.setCurrentIndex(target)
        finally:
            self._sim_backend_tabs_guard = False

    def _restore_arm_panels_order(self) -> None:
        """按 eai/arm_panels_order.json 恢复手臂功能子面板顺序。"""
        bar = getattr(self, "arm_panel_tab_bar", None)
        if bar is None or bar.count() <= 1:
            return
        desired = _eai_infrastructure_ui_storage.preferred_arm_panels_order(self._arm_panel_selector_keys())
        if not desired or desired == self._arm_panel_selector_keys():
            return
        self._arm_panel_tabs_order_guard = True
        try:
            for target_idx, key in enumerate(desired):
                cur = -1
                for i in range(bar.count()):
                    if str(bar.tabData(i) or "") == key:
                        cur = i
                        break
                if cur >= 0 and cur != target_idx:
                    bar.moveTab(cur, target_idx)
        finally:
            self._arm_panel_tabs_order_guard = False

    def _sync_arm_panel_tab_from_radios(self) -> None:
        """把隐藏 radio 的选中态同步到手臂功能 TabBar。"""
        bar = getattr(self, "arm_panel_tab_bar", None)
        if bar is None or bar.count() <= 0:
            return
        key = self._arm_panel_selector_key()
        target = -1
        for i in range(bar.count()):
            if str(bar.tabData(i) or "") == key:
                target = i
                break
        if target < 0 or target == bar.currentIndex():
            return
        self._arm_panel_tabs_guard = True
        try:
            bar.setCurrentIndex(target)
        finally:
            self._arm_panel_tabs_guard = False

    def _wrist_color_topics(self, side: str) -> _eai_platform_bootstrap.Tuple[str, ...]:
        return _eai_configuration_policies.WRIST_COLOR_TOPIC_ALIASES.get(side, ())

    def _wrist_depth_topics(self, side: str) -> _eai_platform_bootstrap.Tuple[str, ...]:
        return _eai_configuration_policies.WRIST_DEPTH_TOPIC_ALIASES.get(side, ())

    def _resolve_wrist_color_topic(self, side: str) -> str:
        """优先选当前已有帧 / 已发现的腕部彩色 topic（hand_* 或 left_wrist_*）。"""
        aliases = self._wrist_color_topics(side)
        if not aliases:
            return "/camera/left_wrist_color" if side == "left" else "/camera/right_wrist_color"
        known = set(self._cached_topic_names())
        for t in aliases:
            if self._frame_cache.get(t) is not None:
                return t
            panel = self.panels.get(t)
            if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and getattr(panel, "_latest_image", None) is not None:
                return t
        for t in aliases:
            if t in known or t in self.topic_checks:
                return t
        return aliases[0]

    def _ensure_mv6d_camera_feeds(self) -> None:
        """多视角 6D：仅确保头/手彩色在列表中；不强制勾选 depth/腕部（会卡死 UI）。"""
        img_types = ["sensor_msgs/msg/Image"]
        base = dict(getattr(self, "_topic_types", {}) or {})
        seed_colors = [
            "/camera/head_color",
            "/camera/hand_left_color",
            "/camera/hand_right_color",
        ]
        for topic in seed_colors:
            base.setdefault(topic, list(img_types))
        # 只补列表项，不整表重建 / 不强制勾选
        for topic in seed_colors:
            if topic in self.topic_checks:
                continue
            cb = _eai_platform_bootstrap.QCheckBox(f"{topic}  [Image]")
            cb.setChecked(topic not in self._topic_user_unchecked)
            cb.stateChanged.connect(self._on_selection_changed)
            self.topic_checks[topic] = cb
            if hasattr(self, "topic_list_layout"):
                # 插在 stretch 前
                stretch_at = self.topic_list_layout.count() - 1
                if stretch_at < 0:
                    self.topic_list_layout.addWidget(cb)
                else:
                    self.topic_list_layout.insertWidget(max(0, stretch_at), cb)
            self._topic_types[topic] = list(img_types)

        enabled = {t for t, cb in self.topic_checks.items() if cb.isChecked()}
        for topic in seed_colors:
            if topic in self._topic_user_unchecked:
                continue
            cb = self.topic_checks.get(topic)
            if cb is None:
                continue
            if not cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
            enabled.add(topic)
        self._apply_selection(enabled)

    def _apply_arm_panel_stack(self) -> None:
        stack = getattr(self, "_arm_panel_stack", None)
        pages = getattr(self, "_arm_panel_page_by_key", None) or {}
        if stack is None:
            return
        key = self._arm_panel_selector_key()
        page = pages.get(key)
        if page is None:
            return
        idx = stack.indexOf(page)
        if idx >= 0 and idx != stack.currentIndex():
            stack.setCurrentIndex(idx)
        if key == "mv6d":
            # 不再自动强制订阅全部 RGB-D（切 tab / 启动会卡死）。
            # 仅轻量刷预览；真正要深度时在「估计一次/实时估计」里再 ensure。
            _eai_platform_bootstrap.QTimer.singleShot(0, self._refresh_mv6d_live_previews_for_pick)

    def _update_ctx_ui(self, *_args) -> None:
        if not hasattr(self, "ctx_sync_start_btn"):
            return
        node_running = self._replay_launcher.is_running()
        has_video = bool((self._ctx_video_path or "").strip()) and _eai_platform_bootstrap.os.path.isfile(
            self._ctx_video_path or ""
        )
        has_rrd = bool((self._ctx_rrd_path or "").strip()) and _eai_platform_bootstrap.os.path.isfile(
            self._ctx_rrd_path or ""
        )
        robot_blockers = self.node.get_replay_prerequisite_warnings()
        can_sync = (
            (not node_running)
            and has_video
            and has_rrd
            and not robot_blockers
        )
        self.ctx_select_video_btn.setEnabled(not node_running)
        self.ctx_select_rrd_btn.setEnabled(not node_running)
        self.ctx_rerun_btn.setEnabled(has_rrd)
        self.ctx_loop_spin.setEnabled(not node_running)
        self.ctx_sync_start_btn.setEnabled(can_sync)
        if can_sync:
            self.ctx_sync_start_btn.setToolTip(
                "同步播放演示视频与 RRD 回放（先决条件已满足）。"
            )
        else:
            reasons: _eai_platform_bootstrap.List[str] = []
            if node_running:
                reasons.append("回放节点已在运行")
            if not has_video:
                reasons.append("请先选择演示视频")
            if not has_rrd:
                reasons.append("请先选择配对 .rrd")
            reasons.extend(robot_blockers)
            self.ctx_sync_start_btn.setToolTip(
                "不可同步开始:\n" + "\n".join(f"• {r}" for r in reasons)
            )
        self.ctx_stop_btn.setEnabled(
            node_running
            or (
                getattr(self, "ctx_video_player", None) is not None
                and self.ctx_video_player.is_playing()
            )
        )
        if node_running:
            label = self.node.get_replay_state_label()
            completed, total = self._replay_launcher.get_loop_progress()
            loop_hint = f" [{min(completed + 1, total)}/{total}]" if total > 1 else ""
            self.ctx_status_label.setText(f"状态: 同步中 · {label}{loop_hint}")
        elif has_video and has_rrd and not robot_blockers:
            self.ctx_status_label.setText("状态: 已就绪（可同步开始）")
        elif has_video and has_rrd:
            self.ctx_status_label.setText("状态: 文件已选（机器人先决条件未满足）")
        elif has_video:
            self.ctx_status_label.setText("状态: 已选视频（需配对 RRD）")
        elif has_rrd:
            self.ctx_status_label.setText("状态: 已选 RRD（需选视频）")
        else:
            self.ctx_status_label.setText("状态: 未开始")

    def _update_replay_rrd_path_display(self, path: str) -> None:
        text = (path or "").strip()
        if text:
            self.replay_rrd_path_edit.setText(text)
            self.replay_rrd_path_edit.setToolTip(text)
        else:
            self.replay_rrd_path_edit.clear()
            self.replay_rrd_path_edit.setToolTip("当前已选择、用于回放的 .rrd 文件完整路径")

    def _update_replay_ui(self, *_args) -> None:
        state = self.node.get_replay_state()
        label = self.node.get_replay_state_label()
        node_running = self._replay_launcher.is_running()
        completed, total = self._replay_launcher.get_loop_progress()
        loop_hint = ""
        if node_running and total > 1:
            current = min(completed + 1, total) if state in (0, 1, 2) else completed
            loop_hint = f" [{current}/{total}]"
        if node_running:
            self.replay_status_label.setText(f"回放: {label}{loop_hint}")
        else:
            self.replay_status_label.setText(f"回放: {label}")

        path = self._replay_launcher.current_rrd_path() or self._selected_rrd_path
        self._update_replay_rrd_path_display(path)

        has_path = bool((self._selected_rrd_path or "").strip())
        robot_blockers = self.node.get_replay_prerequisite_warnings()
        can_start = (not node_running) and has_path and not robot_blockers
        self.replay_count_spin.setEnabled(not node_running)
        self.replay_select_btn.setEnabled(not node_running)
        self.replay_start_btn.setEnabled(can_start)
        if can_start:
            self.replay_start_btn.setToolTip(
                "启动 rrd_replay 并开始回放（先决条件已满足）。"
            )
        else:
            reasons: _eai_platform_bootstrap.List[str] = []
            if node_running:
                reasons.append("回放节点已在运行")
            if not has_path:
                reasons.append("请先选择 .rrd 文件")
            reasons.extend(robot_blockers)
            self.replay_start_btn.setToolTip(
                "不可开始回放:\n" + "\n".join(f"• {r}" for r in reasons)
            )
        self.replay_stop_btn.setEnabled(node_running)
        self._update_ctx_ui()

        if state == 1:
            color = "#50fa7b"
        elif state in (2, 0) and node_running:
            color = "#ffb86c"
        elif state in (3, 4):
            color = _eai_configuration_policies.UI_TEXT_MUTED
        else:
            color = "#7ec8ff"
        self.replay_status_label.setStyleSheet(f"color: {color};")

    def _append_lingbot_log(self, line: str) -> None:
        if not hasattr(self, "lingbot_log_edit"):
            return
        self.lingbot_log_edit.append(line)
        bar = self.lingbot_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_lingbot_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_lingbot_launcher", None)
            and self._lingbot_launcher.is_running()
        )
        self.lingbot_run_btn.setEnabled(not running)
        self.lingbot_stop_btn.setEnabled(running)
        self.lingbot_variant_combo.setEnabled(not running)
        self.lingbot_source_combo.setEnabled(not running)
        self.lingbot_size_spin.setEnabled(not running)
        file_mode = str(self.lingbot_source_combo.currentData() or "") == "file"
        self.lingbot_path_edit.setEnabled(not running and file_mode)
        self.lingbot_browse_btn.setEnabled(not running and file_mode)
        self.lingbot_ckpt_edit.setEnabled(not running)
        self.lingbot_ckpt_browse_btn.setEnabled(not running)
        if running:
            self.lingbot_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.lingbot_status_label.setStyleSheet("")

    def _resolve_lingbot_input_image(self) -> _eai_platform_bootstrap.Optional[str]:
        source = str(self.lingbot_source_combo.currentData() or "camera")
        if source == "camera":
            picked = self._pick_sam3_color_source()
            if picked is None:
                self._append_lingbot_log(
                    "[ERROR] 无可用彩色图像，请勾选 color topic 或改用示例图/文件"
                )
                self._on_lingbot_status("无可用相机图像")
                return None
            topic, image, _panel = picked
            out_dir = _eai_configuration_tabs.LINGBOT_VISION_CACHE_DIR
            _eai_platform_bootstrap.os.makedirs(out_dir, exist_ok=True)
            path = _eai_platform_bootstrap.os.path.join(out_dir, "input_camera.png")
            rgb = image
            if rgb.ndim == 2:
                rgb = _eai_platform_bootstrap.cv2.cvtColor(rgb, _eai_platform_bootstrap.cv2.COLOR_GRAY2BGR)
            _eai_platform_bootstrap.cv2.imwrite(path, rgb)
            self._append_lingbot_log(f"已从 {topic} 抓帧 -> {path}")
            return path
        if source == "example":
            path = _eai_configuration_tabs.LINGBOT_VISION_EXAMPLE
        else:
            path = _eai_platform_bootstrap.os.path.abspath(
                _eai_platform_bootstrap.os.path.expanduser(self.lingbot_path_edit.text().strip())
            )
        if not _eai_platform_bootstrap.os.path.isfile(path):
            self._append_lingbot_log(f"[ERROR] 输入图像不存在: {path}")
            self._on_lingbot_status("输入图像不存在")
            return None
        return path

    def _append_lingbot_depth_log(self, line: str) -> None:
        if not hasattr(self, "lingbot_depth_log_edit"):
            return
        self.lingbot_depth_log_edit.append(line)
        bar = self.lingbot_depth_log_edit.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _update_lingbot_depth_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_lingbot_depth_launcher", None)
            and self._lingbot_depth_launcher.is_running()
        )
        running = running or bool(getattr(self, "_mv6d_lingbot_active", False))
        source = str(self.lingbot_depth_source_combo.currentData() or "camera")
        self.lingbot_depth_topic_combo.setEnabled(not running and source == "camera")
        self.lingbot_depth_topic_refresh_btn.setEnabled(not running and source == "camera")
        file_mode = source == "file"
        example_mode = source == "example"
        self.lingbot_depth_run_btn.setEnabled(not running)
        self.lingbot_depth_stop_btn.setEnabled(running)
        self.lingbot_depth_model_combo.setEnabled(not running)
        self.lingbot_depth_source_combo.setEnabled(not running)
        self.lingbot_depth_device_combo.setEnabled(not running)
        self.lingbot_depth_example_spin.setEnabled(not running and example_mode)
        self.lingbot_depth_model_edit.setEnabled(not running)
        self.lingbot_depth_model_browse_btn.setEnabled(not running)
        for w in (
            self.lingbot_depth_rgb_edit,
            self.lingbot_depth_rgb_browse_btn,
            self.lingbot_depth_depth_edit,
            self.lingbot_depth_depth_browse_btn,
            self.lingbot_depth_intrinsics_edit,
            self.lingbot_depth_intrinsics_browse_btn,
        ):
            w.setEnabled(not running and file_mode)
        if running:
            self.lingbot_depth_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        else:
            self.lingbot_depth_status_label.setStyleSheet("")

    def _list_preview_color_topics(self) -> list[str]:
        """图像预览中可用的彩色 topic（有最新帧优先）。"""
        topics: list[str] = []
        seen: set[str] = set()
        for topic, panel in sorted(self.panels.items()):
            if _eai_perception_image_geometry.is_depth_topic(topic) or not isinstance(panel, _eai_ui_cameras_panels.CameraPanel):
                continue
            if panel._latest_image is None:
                continue
            topics.append(topic)
            seen.add(topic)
        for topic in sorted(self._frame_cache.keys()):
            if topic in seen or _eai_perception_image_geometry.is_depth_topic(topic):
                continue
            image = self._frame_cache.get(topic)
            if image is None or _eai_platform_bootstrap.np.asarray(image).ndim < 2:
                continue
            topics.append(topic)
            seen.add(topic)
        for topic in sorted(self.panels.keys()):
            if topic in seen or _eai_perception_image_geometry.is_depth_topic(topic):
                continue
            if isinstance(self.panels.get(topic), _eai_ui_cameras_panels.CameraPanel):
                topics.append(topic)
                seen.add(topic)
        return topics

    def _list_lingbot_depth_preview_topics(self) -> list[str]:
        return self._list_preview_color_topics()

    def _get_preview_color_frame(
        self, topic: str
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Optional[_eai_ui_cameras_panels.CameraPanel]]]:
        panel = self.panels.get(topic)
        if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) and panel._latest_image is not None:
            return _eai_platform_bootstrap.np.asarray(panel._latest_image).copy(), panel
        image = self._frame_cache.get(topic)
        if image is not None and _eai_platform_bootstrap.np.asarray(image).ndim >= 2:
            return _eai_platform_bootstrap.np.asarray(image).copy(), (
                panel if isinstance(panel, _eai_ui_cameras_panels.CameraPanel) else None
            )
        return None

    def _get_lingbot_depth_color_frame(
        self, topic: str
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Optional[_eai_ui_cameras_panels.CameraPanel]]]:
        return self._get_preview_color_frame(topic)

    def _pick_preview_color_source(
        self, combo: _eai_platform_bootstrap.Optional[object] = None
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Optional[_eai_ui_cameras_panels.CameraPanel]]]:
        topic = ""
        if combo is not None:
            try:
                topic = str(combo.currentData() or "").strip()
            except Exception:
                topic = ""
        if not topic:
            topic = (self._last_sam3_topic or "").strip()
        if topic and not _eai_perception_image_geometry.is_depth_topic(topic):
            got = self._get_preview_color_frame(topic)
            if got is not None:
                color_bgr, panel = got
                return topic, color_bgr, panel
        return self._pick_sam3_color_source()

    def _pick_lingbot_depth_color_source(
        self,
    ) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[str, _eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.Optional[_eai_ui_cameras_panels.CameraPanel]]]:
        combo = getattr(self, "lingbot_depth_topic_combo", None)
        return self._pick_preview_color_source(combo)

    def _refresh_preview_topic_combo(
        self, combo: object, prefer: _eai_platform_bootstrap.Optional[str] = None
    ) -> None:
        if combo is None:
            return
        topics = self._list_preview_color_topics()
        current = prefer or str(combo.currentData() or "")
        if not current:
            current = (self._last_sam3_topic or "").strip()
        combo.blockSignals(True)
        combo.clear()
        if not topics:
            combo.addItem("（无预览图像）", "")
        else:
            for topic in topics:
                combo.addItem(topic, topic)
            if current and current in topics:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            elif current and current not in topics:
                combo.addItem(current, current)
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        combo.blockSignals(False)
        self._retry_pending_combo_ui_state()

