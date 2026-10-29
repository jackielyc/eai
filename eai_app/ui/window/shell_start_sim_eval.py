"""ui.window.shell start sim eval — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_adapters_simulation_robodojo = module_ref('eai_app.adapters.simulation.robodojo')
_eai_adapters_simulation_robodojo_discovery = module_ref('eai_app.adapters.simulation.robodojo_discovery')
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ShellStartSimEvalMixin:
    def _start_sim_eval(self, *, viewer_only: bool = False) -> None:
        if getattr(self, "_mujoco_launcher", None) and self._mujoco_launcher.is_running():
            self._append_sim_log("MuJoCo / MolmoSpaces 正在运行，请先停止后再启动 Isaac 评测")
            return
        if self._sim_backend_id() != "isaac":
            self._append_sim_log("请先将仿真后端切换为「Isaac / RoboDojo」")
            return
        self._set_sim_paused(False, log=False)
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
        # currentData() 为 "" 表示「无」：勿回落到 demo_policy
        if viewer_only:
            policy_dir = ""
            skip_policy = True
            use_gui = True
            self._append_sim_log(
                "[只开界面] 策略=无，强制开仿真 GUI，不部署/不调用模型"
            )
        else:
            policy_data = self.sim_eval_policy_combo.currentData()
            if policy_data is None:
                policy_dir = "XPolicyLab/policy/demo_policy"
            else:
                policy_dir = str(policy_data)
            skip_policy = not policy_dir.strip()
            use_gui = bool(
                getattr(self, "sim_eval_use_gui_check", None) is not None
                and self.sim_eval_use_gui_check.isChecked()
            )
            # Pi_05 must use OpenPI ckpt — replace LingBot_VA leftovers in-place.
            ckpt_preview = self.sim_eval_ckpt_edit.text().strip() or "demo"
            if (not skip_policy) and _eai_configuration_policies._is_pi05_robodojo_policy(policy_dir) and (
                ckpt_preview == "demo"
                or "RoboDojo-cotrain" in ckpt_preview
                or _eai_configuration_policies._ckpt_owned_by_lingbot_va_not_pi05(ckpt_preview)
            ):
                fixed = _eai_configuration_policies._default_pi05_ckpt_text()
                self.sim_eval_ckpt_edit.setText(fixed)
                act_idx = self.sim_eval_action_combo.findData("joint")
                if act_idx >= 0:
                    self.sim_eval_action_combo.setCurrentIndex(act_idx)
                self._append_sim_log(
                    f"[策略] Pi_05 不使用 LingBot_VA ckpt={ckpt_preview}；"
                    f"已改为 OpenPI 权重 {fixed}"
                )
        policy_env = _eai_configuration_policies.ROBODOJO_ENV_DEFAULT
        if (not skip_policy) and "starVLA" in policy_dir and _eai_platform_bootstrap.os.path.isdir(
            _eai_configuration_policies.ROBODOJO_STARVLA_ENV_DEFAULT
        ):
            policy_env = _eai_configuration_policies.ROBODOJO_STARVLA_ENV_DEFAULT
        elif (not skip_policy) and _eai_configuration_models.is_lingbot_va_policy(policy_dir):
            va_env = _eai_configuration_models.resolve_lingbot_va_conda_env(_eai_configuration_policies.ROBODOJO_LINGBOT_VA_ENV_DEFAULT)
            if va_env and _eai_platform_bootstrap.os.path.isdir(va_env):
                policy_env = va_env
        elif (not skip_policy) and _eai_configuration_policies._is_pi05_robodojo_policy(policy_dir):
            if _eai_platform_bootstrap.os.path.isdir(_eai_configuration_policies.ROBODOJO_OPENPI_ENV_DEFAULT):
                policy_env = _eai_configuration_policies.ROBODOJO_OPENPI_ENV_DEFAULT
        elif (not skip_policy) and _eai_configuration_models.is_vpp2_policy(policy_dir):
            vpp2_env = _eai_configuration_models.resolve_vpp2_conda_env(_eai_configuration_policies.ROBODOJO_VPP2_ENV_DEFAULT)
            if vpp2_env and _eai_platform_bootstrap.os.path.isdir(vpp2_env):
                policy_env = vpp2_env
            try:
                _eai_configuration_models.ensure_vpp2_adapter(robodojo_root=_eai_configuration_policies.ROBODOJO_ROOT_DEFAULT)
            except Exception:
                pass
        # 一体启动：评测进程内先起策略再起 Isaac（旧 robodojo.sh eval）；不依赖「部署启动」
        bundled = bool(
            (not skip_policy)
            and getattr(self, "sim_eval_bundled_check", None) is not None
            and self.sim_eval_bundled_check.isChecked()
        )
        policy_port = (
            _eai_configuration_models.DEFAULT_VPP2_PORT
            if (not skip_policy) and _eai_configuration_models.is_vpp2_policy(policy_dir)
            else _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
        )
        start_kwargs = dict(
            bridge_dir=bridge_dir,
            policy_dir=policy_dir,
            ckpt=self.sim_eval_ckpt_edit.text().strip() or "demo",
            action_type=str(self.sim_eval_action_combo.currentData() or "ee"),
            policy_env=policy_env,
            eval_env=_eai_configuration_policies.ROBODOJO_ENV_DEFAULT,
            robodojo_root=_eai_configuration_policies.ROBODOJO_ROOT_DEFAULT,
            display=_eai_platform_bootstrap.os.environ.get("DISPLAY", ":1.0"),
            use_gui=use_gui,
            reuse_existing=True,
            attach_external=False,
            policy_host=_eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT,
            policy_port=policy_port,
        )

        # 已由本窗口管理：热切换（不拦策略检查）
        if self._sim_eval_launcher.is_running():
            result = self._sim_eval_launcher.start(task, **start_kwargs)
            if result in ("switched", "reused"):
                self._append_sim_log(
                    f"{'热切换' if result == 'switched' else '保持'}任务 → {task}"
                    "（复用已运行仿真）"
                )
                self._activate_sim_camera_preview(force=False)
                self._update_sim_eval_run_ui()
                if (not viewer_only) and self._sim_rynnvalue_hud_enabled():
                    if self._ensure_live_hud_reward_or_abort("isaac"):
                        self._start_isaac_rynnvalue_npy_hud(bridge_dir)
                return

        # 本窗口未管理，但系统里可能已有 Isaac / eval_client → 先复用
        existing = _eai_adapters_simulation_robodojo.RoboDojoEvalLauncher.list_running_eval_clients()
        if existing:
            self._append_sim_log(
                "检测到已运行仿真: "
                + ", ".join(
                    f"pid={c['pid']} task={c.get('task') or '?'}" for c in existing
                )
            )
            result = self._sim_eval_launcher.start(task, **start_kwargs)
            if result in ("reused", "switched"):
                self._sim_preview_wait_logged = False
                self._activate_sim_camera_preview(force=False)
                self._update_sim_eval_run_ui()
                self._append_sim_log(
                    f"已复用仿真进程（{result}），任务目标={task}；未重头启动 Isaac"
                )
                if (not viewer_only) and self._sim_rynnvalue_hud_enabled():
                    if self._ensure_live_hud_reward_or_abort("isaac"):
                        self._start_isaac_rynnvalue_npy_hud(bridge_dir)
                return
            self._append_sim_log("复用失败，将尝试新启动评测…")

        if not skip_policy and bundled:
            # 一体启动会自己拉起 server，须先释放端口上的独立部署。
            host = _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT
            port = _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
            if (
                hasattr(self, "_robodojo_policy_launcher")
                and self._robodojo_policy_launcher.is_running()
            ):
                self._append_sim_log(
                    "[策略] 一体启动：先停止本窗口独立「部署」进程，改由评测内拉起"
                )
                self._robodojo_policy_launcher.stop()
                self._update_sim_policy_deploy_ui()
                # stop 异步；等端口释放再继续
                import time as _time

                for _ in range(30):
                    info_wait = _eai_adapters_simulation_robodojo_discovery.inspect_robodojo_policy_listener(host, port)
                    if not info_wait.get("listening"):
                        break
                    _time.sleep(0.2)
            info = _eai_adapters_simulation_robodojo_discovery.inspect_robodojo_policy_listener(host, port)
            if info.get("listening"):
                msg = (
                    f"[策略] 一体启动需要空闲端口 {host}:{port}，"
                    f"当前仍有进程占用"
                    + (
                        f"（{info.get('policy_name') or '?'}/"
                        f"{info.get('ckpt_name') or '?'}）"
                        if info.get("is_policy_server")
                        else ""
                    )
                    + "。请先「部署停止」或结束占用后再一体启动。"
                )
                self._append_sim_log(msg)
                try:
                    from PyQt5.QtWidgets import QMessageBox

                    QMessageBox.warning(self, "端口被占用", msg.replace("[策略] ", ""))
                except Exception:
                    pass
                return
            start_kwargs["attach_external"] = False
            self._append_sim_log(
                f"[策略] 一体启动：评测内先起 "
                f"{_eai_platform_bootstrap.os.path.basename(policy_dir.rstrip('/'))}/"
                f"{start_kwargs['ckpt']} action={start_kwargs['action_type']}，"
                f"就绪后再起 Isaac（旧流程，利于避 OOM）"
            )
        elif not skip_policy:
            # 分离模式：必须已「部署启动」且匹配；评测绝不顺带起 server。
            ckpt = str(start_kwargs["ckpt"])
            action_type = str(start_kwargs.get("action_type") or "ee")
            info = _eai_adapters_simulation_robodojo_discovery.inspect_robodojo_policy_listener(
                _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT, _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
            )
            if not _eai_adapters_simulation_robodojo_discovery.robodojo_policy_matches_listener(
                info,
                policy_dir=policy_dir,
                ckpt=ckpt,
                action_type=action_type,
            ):
                host = _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT
                port = _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
                want = (
                    f"{_eai_platform_bootstrap.os.path.basename(policy_dir.rstrip('/'))}/{ckpt}"
                    f" action={action_type}"
                )
                if not info.get("listening"):
                    msg = (
                        f"[策略] 模型未部署（{host}:{port} 无服务）。"
                        f"请先对「{want}」点击「部署启动」，就绪后再「启动评测」；"
                        "或勾选「一体启动」由评测内先起模型。"
                    )
                elif info.get("is_policy_server"):
                    got = (
                        f"{info.get('policy_name') or '?'}/"
                        f"{info.get('ckpt_name') or '?'}"
                        f" action={info.get('action_type') or '?'}"
                    )
                    msg = (
                        f"[策略] 已部署的是 {got}，与当前选择 {want} 不一致。"
                        "请先「部署停止」再部署目标模型，或改选与线上一致的策略/ckpt/action。"
                        "评测不会自动拉起策略服务（除非勾选「一体启动」）。"
                    )
                else:
                    msg = (
                        f"[策略] {host}:{port} 被占用但不是策略服务。"
                        "请先释放端口并「部署启动」目标模型，再启动评测。"
                    )
                self._append_sim_log(msg)
                try:
                    from PyQt5.QtWidgets import QMessageBox

                    QMessageBox.warning(self, "需要先部署策略", msg.replace("[策略] ", ""))
                except Exception:
                    pass
                return
            start_kwargs["attach_external"] = True
            self._append_sim_log(
                f"[策略] 已确认部署就绪 "
                f"{_eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT}:{_eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT} "
                f"({info.get('policy_name') or _eai_platform_bootstrap.os.path.basename(policy_dir)}/"
                f"{info.get('ckpt_name') or ckpt}"
                f" action={info.get('action_type') or action_type})，评测只起 client"
            )

        # 全新启动：清旧帧并按需起相机桥
        self._reset_sim_bridge_frames()
        self._sim_preview_wait_logged = False
        self._sim_eval_auto_bridge = False
        if use_gui:
            if not self._sim_bridge_launcher.is_running():
                self._sim_bridge_launcher.start(
                    bridge_dir,
                    hz=float(self.sim_bridge_hz_spin.value()),
                )
                self._sim_eval_auto_bridge = True
                self._update_sim_bridge_ui()
            else:
                self._sim_eval_auto_bridge = False
        elif self._sim_bridge_launcher.is_running():
            self._append_sim_log("headless：停止已在运行的相机桥（预览改直读 .npy）")
            self._sim_bridge_launcher.stop()
            self._sim_eval_auto_bridge = False
            self._update_sim_bridge_ui()
        mode_note = "GUI（仿真界面）" if use_gui else "headless（无 Isaac 窗口）"
        if skip_policy:
            self._append_sim_log(
                f"评测模式: {mode_note}；策略=无（只启评测，不启策略服务）；"
                f"EVAL_NUM=keep 常驻；共享帧={bridge_dir}"
            )
        elif start_kwargs.get("attach_external"):
            self._append_sim_log(
                f"评测模式: {mode_note}；复用已部署策略 "
                f"{start_kwargs['policy_host']}:{start_kwargs['policy_port']}；"
                f"EVAL_NUM=keep 常驻；共享帧={bridge_dir}"
            )
        else:
            self._append_sim_log(
                f"评测模式: {mode_note}；一体启动（先策略后仿真）；"
                f"EVAL_NUM=keep 常驻；共享帧={bridge_dir}"
            )
        result = self._sim_eval_launcher.start(task, **start_kwargs)
        self._update_sim_eval_run_ui()
        self._activate_sim_camera_preview()
        self._append_sim_log(
            f"启动结果={result}；图像预览 RGB-D："
            "/camera/{left_wrist,head,right_wrist}_{color,depth}"
            "（优先共享 .npy，忽略同名真机 ROS 流）"
        )
        # RynnValue Live HUD: poll cam_head.npy written by Isaac cam bridge.
        if (not viewer_only) and self._sim_rynnvalue_hud_enabled():
            if not self._ensure_live_hud_reward_or_abort("isaac"):
                return
            self._start_isaac_rynnvalue_npy_hud(bridge_dir)

    def _update_sim_policy_deploy_ui(self, *_args) -> None:
        running = bool(
            getattr(self, "_robodojo_policy_launcher", None)
            and self._robodojo_policy_launcher.is_running()
        )
        info = _eai_adapters_simulation_robodojo_discovery.inspect_robodojo_policy_listener(
            _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT, _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
        )
        listening = bool(info.get("listening"))
        healthy = bool(info.get("is_policy_server"))
        # 进程常驻 = 正常；端口已开就视为就绪，不要一直「部署中」。
        if (
            running
            and listening
            and healthy
            and hasattr(self, "_sim_policy_ready_timer")
            and self._sim_policy_ready_timer.isActive()
        ):
            self._sim_policy_ready_timer.stop()
        if hasattr(self, "sim_policy_deploy_start_btn"):
            # Allow click even when listening — handler will skip if already OK.
            self.sim_policy_deploy_start_btn.setEnabled(not running)
        if hasattr(self, "sim_policy_deploy_stop_btn"):
            self.sim_policy_deploy_stop_btn.setEnabled(running or (listening and healthy))
        if not hasattr(self, "sim_policy_deploy_status"):
            return
        host = _eai_configuration_policies.ROBODOJO_POLICY_HOST_DEFAULT
        port = _eai_configuration_policies.ROBODOJO_POLICY_PORT_DEFAULT
        pol = str(info.get("policy_name") or "")
        ck = str(info.get("ckpt_name") or "")
        act = str(info.get("action_type") or "")
        detail = f"{pol}" + (f"/{ck}" if ck else "") + (f" action={act}" if act else "")
        if listening and healthy:
            owned = "本窗口托管" if running else "可复用"
            self.sim_policy_deploy_status.setText(
                f"策略: 已就绪 {host}:{port}"
                + (f" ({detail})" if detail.strip("/") else "")
                + f" — {owned}"
            )
            self.sim_policy_deploy_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
        elif running:
            self.sim_policy_deploy_status.setText(
                f"策略: 部署中（等待端口）→ {host}:{port}"
            )
            self.sim_policy_deploy_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        elif listening:
            self.sim_policy_deploy_status.setText(
                f"策略: 端口占用但非策略服务 {host}:{port}"
            )
            self.sim_policy_deploy_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_ORANGE};")
        else:
            self.sim_policy_deploy_status.setText("策略: 未部署")
            self.sim_policy_deploy_status.setStyleSheet("")

    def _set_sim_paused(self, paused: bool, *, log: bool = True) -> bool:
        """Write shared sim_control.json and sync both pause buttons."""
        self._sim_paused_ui = bool(paused)
        ok = True
        if _eai_configuration_policies.write_sim_control is not None:
            try:
                ok = bool(
                    _eai_configuration_policies.write_sim_control(
                        root=self._sim_bridge_dir(),
                        paused=bool(paused),
                        merge=True,
                    )
                )
            except Exception as exc:
                ok = False
                if log:
                    self._append_sim_log(f"[暂停] 写入失败: {exc}")
        elif _eai_configuration_policies.clear_sim_control is not None and not paused:
            try:
                _eai_configuration_policies.clear_sim_control(self._sim_bridge_dir())
            except Exception:
                pass
        self._update_sim_pause_ui()
        if log:
            msg = "已暂停任务" if paused else "已继续任务"
            try:
                self._append_sim_log(f"[暂停] {msg}")
            except Exception:
                pass
            try:
                self._append_mujoco_log(f"[暂停] {msg}")
            except Exception:
                pass
        return ok

    def _update_sim_pause_ui(self) -> None:
        running = self._any_sim_process_running()
        # Isaac may be an externally attached session (stop btn still enabled).
        try:
            ext_isaac = bool(
                _eai_adapters_simulation_robodojo.RoboDojoEvalLauncher.isaac_sim_status().get("running")
            )
        except Exception:
            ext_isaac = False
        can_pause = bool(running or ext_isaac)
        paused = self._sim_paused_state() if can_pause else False
        if not can_pause:
            self._sim_paused_ui = False
        label = "继续" if paused else "暂停"
        tip = (
            "当前已暂停；点击继续运行任务。"
            if paused
            else "暂停当前仿真任务（保留状态，可继续）。"
        )
        for attr in ("sim_eval_pause_btn", "mujoco_pause_btn"):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            btn.blockSignals(True)
            try:
                btn.setEnabled(can_pause)
                btn.setChecked(paused and can_pause)
                btn.setText(label)
                btn.setToolTip(tip)
            finally:
                btn.blockSignals(False)

    def _refresh_sim_eval_tree(self) -> None:
        root = self._sim_eval_root_path()
        self.sim_eval_tree.clear()
        self.sim_eval_detail_edit.clear()
        self.sim_eval_files_list.clear()
        if not _eai_platform_bootstrap.os.path.isdir(root):
            self.sim_eval_detail_edit.setPlainText(f"目录不存在:\n{root}")
            return
        filter_text = self.sim_eval_filter_edit.text().strip().lower()
        try:
            names = sorted(_eai_platform_bootstrap.os.listdir(root))
        except OSError as exc:
            self.sim_eval_detail_edit.setPlainText(f"无法读取目录: {exc}")
            return
        for name in names:
            if name.startswith("."):
                continue
            path = _eai_platform_bootstrap.os.path.join(root, name)
            if not _eai_platform_bootstrap.os.path.isdir(path):
                continue
            if filter_text and filter_text not in name.lower():
                continue
            item = _eai_platform_bootstrap.QTreeWidgetItem([name])
            item.setData(0, _eai_platform_bootstrap.Qt.UserRole, path)
            item.setData(0, _eai_platform_bootstrap.Qt.UserRole + 1, "dir")
            item.addChild(_eai_platform_bootstrap.QTreeWidgetItem(["…"]))
            self.sim_eval_tree.addTopLevelItem(item)
        if getattr(self, "status_bar", None) is not None:
            self.status_bar.showMessage(
                f"评测结果: {self.sim_eval_tree.topLevelItemCount()} 个任务 @ {root}", 4000
            )
        if self._sim_eval_outcome_keyword():
            self._refresh_sim_eval_mp4_by_outcome()

    def _refresh_sim_eval_mp4_by_outcome(self) -> None:
        keyword = self._sim_eval_outcome_keyword()
        self.sim_eval_files_list.clear()
        if not keyword:
            return
        root = self._sim_eval_root_path()
        if not _eai_platform_bootstrap.os.path.isdir(root):
            self.sim_eval_detail_edit.setPlainText(f"目录不存在:\n{root}")
            return
        task_filter = self.sim_eval_filter_edit.text().strip().lower()
        label = str(self.sim_eval_outcome_combo.currentText() or keyword)
        if getattr(self, "status_bar", None) is not None:
            self.status_bar.showMessage(f"正在扫描 {label} 视频…", 0)
        _eai_platform_bootstrap.QApplication.processEvents()
        matches: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Tuple[str, str]] = []  # (display, abs_path)
        try:
            task_names = sorted(_eai_platform_bootstrap.os.listdir(root))
        except OSError as exc:
            self.sim_eval_detail_edit.setPlainText(f"无法读取目录: {exc}")
            return
        for task in task_names:
            if task.startswith("."):
                continue
            task_path = _eai_platform_bootstrap.os.path.join(root, task)
            if not _eai_platform_bootstrap.os.path.isdir(task_path):
                continue
            if task_filter and task_filter not in task.lower():
                continue
            for dirpath, _dirnames, filenames in _eai_platform_bootstrap.os.walk(task_path):
                for name in filenames:
                    lower = name.lower()
                    if not lower.endswith((".mp4", ".avi", ".mkv", ".webm")):
                        continue
                    if keyword not in lower:
                        continue
                    abs_path = _eai_platform_bootstrap.os.path.join(dirpath, name)
                    rel = _eai_platform_bootstrap.os.path.relpath(abs_path, root)
                    matches.append((rel, abs_path))
        matches.sort(key=lambda x: x[0].lower())
        for display, abs_path in matches:
            item = _eai_platform_bootstrap.QListWidgetItem(display)
            item.setData(_eai_platform_bootstrap.Qt.UserRole, abs_path)
            item.setToolTip(abs_path)
            self.sim_eval_files_list.addItem(item)
        if self.sim_eval_files_list.count() > 0:
            self.sim_eval_files_list.setCurrentRow(0)
        summary = (
            f"结果筛选: {label}（文件名含「{keyword}」）\n"
            f"任务过滤: {task_filter or '(无)'}\n"
            f"共 {len(matches)} 个视频\n"
            f"根目录: {root}"
        )
        self.sim_eval_detail_edit.setPlainText(summary)
        if getattr(self, "status_bar", None) is not None:
            self.status_bar.showMessage(f"{label}: 找到 {len(matches)} 个视频", 5000)

    def _fill_sim_eval_children(self, item: _eai_platform_bootstrap.QTreeWidgetItem) -> None:
        path = str(item.data(0, _eai_platform_bootstrap.Qt.UserRole) or "")
        if not path or not _eai_platform_bootstrap.os.path.isdir(path):
            return
        try:
            entries = sorted(_eai_platform_bootstrap.os.listdir(path))
        except OSError:
            return
        for name in entries:
            if name.startswith("."):
                continue
            child_path = _eai_platform_bootstrap.os.path.join(path, name)
            if not _eai_platform_bootstrap.os.path.isdir(child_path):
                continue
            result_json = _eai_platform_bootstrap.os.path.join(child_path, "_result.json")
            child = _eai_platform_bootstrap.QTreeWidgetItem([name])
            child.setData(0, _eai_platform_bootstrap.Qt.UserRole, child_path)
            if _eai_platform_bootstrap.os.path.isfile(result_json):
                rate = self._sim_eval_read_rate(result_json)
                child.setText(0, f"{name}  (sr={rate})")
                child.setData(0, _eai_platform_bootstrap.Qt.UserRole + 1, "run")
            else:
                child.setData(0, _eai_platform_bootstrap.Qt.UserRole + 1, "dir")
                child.addChild(_eai_platform_bootstrap.QTreeWidgetItem(["…"]))
            item.addChild(child)

    def _append_real_eval_log(self, line: str) -> None:
        self.real_eval_log_edit.append(line)
        scrollbar = self.real_eval_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_real_robot_backend_switch_enabled(self, enabled: bool) -> None:
        for radio in self._real_robot_backend_radios():
            radio.setEnabled(enabled)
        bar = getattr(self, "real_robot_backend_tab_bar", None)
        if bar is not None:
            bar.setEnabled(enabled)

    def _restore_real_robot_backends_order(self) -> None:
        """按 eai/real_robot_backends_order.json 恢复真机后端选择条顺序。"""
        bar = getattr(self, "real_robot_backend_tab_bar", None)
        if bar is None or bar.count() <= 1:
            return
        desired = _eai_infrastructure_ui_storage.preferred_real_robot_backends_order(
            self._real_robot_backend_selector_keys()
        )
        if not desired or desired == self._real_robot_backend_selector_keys():
            return
        self._real_robot_tabs_order_guard = True
        try:
            for target_idx, key in enumerate(desired):
                cur = -1
                for i in range(bar.count()):
                    if _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(bar.tabData(i) or "")) == key:
                        cur = i
                        break
                if cur >= 0 and cur != target_idx:
                    bar.moveTab(cur, target_idx)
        finally:
            self._real_robot_tabs_order_guard = False

    def _sync_real_robot_backend_tab_from_radios(self) -> None:
        """把隐藏 radio 的选中态同步到真机后端 TabBar。"""
        bar = getattr(self, "real_robot_backend_tab_bar", None)
        if bar is None or bar.count() <= 0:
            return
        key = self._real_robot_backend_id()
        target = -1
        for i in range(bar.count()):
            if _eai_adapters_robot_launch_utils.normalize_real_robot_backend(str(bar.tabData(i) or "")) == key:
                target = i
                break
        if target < 0 or target == bar.currentIndex():
            return
        self._real_robot_tabs_guard = True
        try:
            bar.setCurrentIndex(target)
        finally:
            self._real_robot_tabs_guard = False

    def _launch_real_robot_stack_detached(self, backend: str) -> None:
        """按后端 shell prefix 直接拉起 robot-service / 臂手栈（不阻塞 UI）。"""
        key = _eai_adapters_robot_launch_utils.normalize_real_robot_backend(backend)
        label = _eai_adapters_robot_launch_utils.real_robot_backend_label(key)
        scripts = _eai_adapters_robot_launch_utils.resolve_real_robot_scripts_dir(key)
        if not scripts:
            self.status_bar.showMessage(f"{label} 未找到 scripts 目录", 5000)
            return
        robot_script = _eai_platform_bootstrap.os.path.join(scripts, _eai_configuration_policies.ROBOT_SERVICE_SCRIPT)
        stack_script = _eai_platform_bootstrap.os.path.join(scripts, _eai_configuration_policies.STACK_SERVICES_SCRIPT)
        prefix = _eai_adapters_robot_launch_utils.build_real_robot_ros_shell_prefix(key)
        if self.node.is_hal_arm_ready() and _eai_platform_bootstrap.os.path.isfile(stack_script):
            script_dir = _eai_platform_bootstrap.os.path.dirname(stack_script)
            cmd = (
                f"{prefix} && cd {script_dir} && "
                f"exec bash {_eai_platform_bootstrap.os.path.basename(stack_script)}"
            )
            if _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", cmd], script_dir):
                self._append_real_eval_log(f"[栈] 已启动 {label} 臂手服务: {stack_script}")
                self.status_bar.showMessage(f"正在启动 {label} 臂手服务栈…", 5000)
            else:
                self.status_bar.showMessage(f"{label} 臂手栈启动失败", 5000)
            return
        if not _eai_platform_bootstrap.os.path.isfile(robot_script):
            self.status_bar.showMessage(f"{label} 缺少 {_eai_configuration_policies.ROBOT_SERVICE_SCRIPT}", 5000)
            return
        script_dir = _eai_platform_bootstrap.os.path.dirname(robot_script)
        cmd = (
            f"{prefix} && cd {script_dir} && "
            f"exec bash {_eai_platform_bootstrap.os.path.basename(robot_script)}"
        )
        if _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", cmd], script_dir):
            self._append_real_eval_log(f"[栈] 已启动 {label} robot-service: {robot_script}")
            self.status_bar.showMessage(f"正在启动 {label} robot-service（约 3 分钟）…", 5000)
        else:
            self.status_bar.showMessage(f"{label} robot-service 启动失败", 5000)

    def _refresh_real_eval_status(self) -> None:
        backend = self._real_robot_backend_id()
        label = _eai_adapters_robot_launch_utils.real_robot_backend_label(backend)
        cams = 0
        try:
            names = list(dict(self.node.get_topic_names_and_types()))
            cams = sum(1 for t in names if t.startswith("/camera") and t.endswith("_color"))
        except Exception:
            names = []
        arm = self.node.get_arm_enable_label()
        hand = (
            f"手: {'已使能' if self.node.is_hand_enabled() else '未使能'}"
        )
        mode = self.node.get_control_mode_label()
        parts = [
            f"后端={label}",
            _eai_adapters_robot_launch_utils.real_robot_stack_ready_hint(backend),
            f"/camera*_color={cams}",
            arm,
            hand,
            mode,
        ]
        blockers = self._real_eval_start_blockers()
        if self._real_eval_active:
            parts.append("评测中")
            self.real_eval_status_label.setText(f"状态: 评测中（{label}）")
            self.real_eval_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};")
            self.real_eval_start_btn.setEnabled(False)
        else:
            ready = not blockers
            self.real_eval_status_label.setText(
                f"状态: {'就绪' if ready else '未就绪'}（{label}）"
            )
            self.real_eval_status_label.setStyleSheet(
                f"color: {_eai_configuration_policies.UI_ACCENT_GREEN};" if ready else ""
            )
            self.real_eval_start_btn.setEnabled(ready)
            if ready:
                self.real_eval_start_btn.setToolTip(
                    "进入真机评测记录。\n"
                    "先决条件已满足：相机 / 手臂使能 / 手部使能。"
                )
            else:
                self.real_eval_start_btn.setToolTip(
                    "不可开始评测:\n" + "\n".join(f"• {w}" for w in blockers)
                )
        self.real_eval_ready_label.setText("  |  ".join(parts))

    def _a2d_ops_group_style(self) -> str:
        return (
            f"QGroupBox {{ color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
            f"margin-top: 6px; padding-top: 4px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
            f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY}; }}"
        )

    def _build_a2d_ops_panel(self) -> _eai_platform_bootstrap.QWidget:
        """真机评测 → A2D：docker / local 两套对齐运维（采集标定 / HIL / FAQ / 视频）。"""
        panel = _eai_platform_bootstrap.QWidget()
        root = _eai_platform_bootstrap.QVBoxLayout(panel)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(6)

        title = _eai_platform_bootstrap.QLabel("A2D 采集标定 · 模型测试 / HIL · 运维")
        title.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold;")
        root.addWidget(title)

        self.a2d_ops_mode_tabs = _eai_platform_bootstrap.QTabWidget()
        self.a2d_ops_mode_tabs.setObjectName("a2dOpsModeTabs")
        self.a2d_ops_mode_tabs.setDocumentMode(True)
        self.a2d_ops_mode_tabs.addTab(
            self._build_a2d_ops_mode_page("docker"), "docker"
        )
        self.a2d_ops_mode_tabs.addTab(
            self._build_a2d_ops_mode_page("local"), "local"
        )
        self.a2d_ops_mode_tabs.setTabToolTip(
            0,
            "依赖 Docker：容器启停 + 必要时 docker exec 进 A2D 跑 ROS",
        )
        self.a2d_ops_mode_tabs.setTabToolTip(
            1,
            "纯本机：不依赖 docker；需本机 a2d-tele install / 同 ROS_DOMAIN",
        )
        root.addWidget(self.a2d_ops_mode_tabs)
        return panel

