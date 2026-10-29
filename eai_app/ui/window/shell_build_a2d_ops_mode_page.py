"""ui.window.shell build a2d ops mode page — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_launch_utils = module_ref('eai_app.adapters.robot.launch_utils')
_eai_adapters_robot_services = module_ref('eai_app.adapters.robot.services')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class ShellBuildA2DOpsModePageMixin:
    def _build_a2d_ops_mode_page(self, mode: str) -> _eai_platform_bootstrap.QWidget:
        """构建 A2D docker 或 local 运维页；功能对齐，仅执行环境不同。"""
        is_local = mode == "local"
        page = _eai_platform_bootstrap.QWidget()
        root = _eai_platform_bootstrap.QVBoxLayout(page)
        root.setContentsMargins(0, 6, 0, 0)
        root.setSpacing(6)

        if is_local:
            tip = _eai_platform_bootstrap.QLabel(
                "本机模式：全部命令在宿主机执行，不调用 docker。"
                "需本机已有 a2d-tele install（或 /opt/psi/rt/a2d-tele）且与现场同 ROS_DOMAIN。"
                " Pixel_err < 3px 为绿。"
            )
        else:
            tip = _eai_platform_bootstrap.QLabel(
                "Docker 模式：按现场文档一键拉起常用步骤；"
                "viewer 已在容器内时 ROS 直接执行，否则可 docker exec；"
                "容器启停需宿主机 CLI（或挂载 docker.sock）。"
                " Pixel_err < 3px 为绿。"
            )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        root.addWidget(tip)

        # —— 环境管理：docker=容器 / local=本机 workspace ——
        if is_local:
            env_box = _eai_platform_bootstrap.QGroupBox("本机环境")
            env_box.setStyleSheet(self._a2d_ops_group_style())
            env_l = _eai_platform_bootstrap.QVBoxLayout(env_box)
            env_l.setContentsMargins(8, 10, 8, 8)
            env_l.setSpacing(6)
            ws_row = _eai_platform_bootstrap.QHBoxLayout()
            ws_row.addWidget(_eai_platform_bootstrap.QLabel("workspace"))
            self.a2d_local_ws_edit = _eai_platform_bootstrap.QLineEdit(_eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir())
            self.a2d_local_ws_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
            self.a2d_local_ws_edit.setToolTip(
                "本机 a2d-tele 根目录（含 install/setup.bash）；"
                "检查环境时读取此路径"
            )
            ws_row.addWidget(self.a2d_local_ws_edit, 1)
            self.a2d_local_doc_btn = _eai_platform_bootstrap.QPushButton("打开文档")
            self.a2d_local_doc_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
            self.a2d_local_doc_btn.setToolTip(_eai_configuration_policies.A2D_OPS_DOC_URL)
            self.a2d_local_doc_btn.clicked.connect(self._on_a2d_open_doc)
            ws_row.addWidget(self.a2d_local_doc_btn)
            env_l.addLayout(ws_row)
            self.a2d_local_env_status = _eai_platform_bootstrap.QLabel("状态: 未检查")
            self.a2d_local_env_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
            self.a2d_local_env_status.setWordWrap(True)
            env_l.addWidget(self.a2d_local_env_status)
            ebtn_row = _eai_platform_bootstrap.QHBoxLayout()
            ebtn_row.setSpacing(6)
            for text, slot, tip_text in (
                ("检查环境", self._on_a2d_local_check_env, "检查 install/setup.bash 与 scripts_pack"),
                ("打开 workspace", self._on_a2d_local_open_ws, "在文件管理器打开 workspace"),
                ("刷新路径", self._on_a2d_local_refresh_ws, "用 resolve_a2d_workspace_dir() 回填"),
            ):
                btn = _eai_platform_bootstrap.QPushButton(text)
                btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
                btn.setToolTip(tip_text)
                btn.clicked.connect(slot)
                ebtn_row.addWidget(btn)
            ebtn_row.addStretch(1)
            env_l.addLayout(ebtn_row)
            root.addWidget(env_box)
        else:
            docker_box = _eai_platform_bootstrap.QGroupBox("容器管理")
            docker_box.setStyleSheet(self._a2d_ops_group_style())
            docker_l = _eai_platform_bootstrap.QVBoxLayout(docker_box)
            docker_l.setContentsMargins(8, 10, 8, 8)
            docker_l.setSpacing(6)
            name_row = _eai_platform_bootstrap.QHBoxLayout()
            name_row.addWidget(_eai_platform_bootstrap.QLabel("容器名"))
            self.a2d_docker_name_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_policies.A2D_DOCKER_CONTAINER_DEFAULT)
            self.a2d_docker_name_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
            self.a2d_docker_name_edit.setToolTip(
                "对应 A2D_DOCKER_CONTAINER / docker ps 名称"
            )
            name_row.addWidget(self.a2d_docker_name_edit, 1)
            self.a2d_doc_btn = _eai_platform_bootstrap.QPushButton("打开文档")
            self.a2d_doc_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
            self.a2d_doc_btn.setToolTip(_eai_configuration_policies.A2D_OPS_DOC_URL)
            self.a2d_doc_btn.clicked.connect(self._on_a2d_open_doc)
            name_row.addWidget(self.a2d_doc_btn)
            docker_l.addLayout(name_row)
            dbtn_row = _eai_platform_bootstrap.QHBoxLayout()
            dbtn_row.setSpacing(6)
            for text, slot, tip_text in (
                ("查看容器", self._on_a2d_docker_ps, "docker ps"),
                ("关闭容器", self._on_a2d_docker_stop, "docker stop <容器>"),
                ("开启容器", self._on_a2d_docker_start, "docker start <容器>"),
                ("重启容器", self._on_a2d_docker_restart, "docker restart <容器>"),
            ):
                btn = _eai_platform_bootstrap.QPushButton(text)
                btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
                btn.setToolTip(tip_text)
                btn.clicked.connect(slot)
                dbtn_row.addWidget(btn)
            dbtn_row.addStretch(1)
            docker_l.addLayout(dbtn_row)
            root.addWidget(docker_box)

        # —— 采集标定 ——
        calib_box = _eai_platform_bootstrap.QGroupBox("采集标定流程")
        calib_box.setStyleSheet(self._a2d_ops_group_style())
        calib_l = _eai_platform_bootstrap.QVBoxLayout(calib_box)
        calib_l.setContentsMargins(8, 10, 8, 8)
        calib_l.setSpacing(6)
        if is_local:
            calib_hint = _eai_platform_bootstrap.QLabel(
                "顺序：tracker 标定 → 启动 GUI → 开启 ROS 服务与脚踏模式。"
                "（本机 a2d-tele / 与现场同 ROS_DOMAIN，不经过 docker）"
            )
            gui_tip = (
                "ros2 launch scripts_pack start_gui.launch.py\n"
                "本机模式：始终在宿主机执行"
            )
        else:
            calib_hint = _eai_platform_bootstrap.QLabel(
                "顺序：tracker 标定 → 启动 GUI → 开启 ROS 服务与脚踏模式。"
                "（需已进入 a2d 容器 / 与现场同 ROS_DOMAIN）"
            )
            gui_tip = (
                "ros2 launch scripts_pack start_gui.launch.py\n"
                "宿主机 viewer 会自动 docker exec 进 A2D 容器执行"
            )
        calib_hint.setWordWrap(True)
        calib_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        calib_l.addWidget(calib_hint)
        cbtn_row = _eai_platform_bootstrap.QHBoxLayout()
        cbtn_row.setSpacing(6)
        tracker_btn = _eai_platform_bootstrap.QPushButton("1. Tracker 标定")
        tracker_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        tracker_btn.setToolTip("ros2 run vr_calibration calibrate_node")
        tracker_btn.clicked.connect(self._on_a2d_tracker_calib)
        cbtn_row.addWidget(tracker_btn)
        gui_btn = _eai_platform_bootstrap.QPushButton("2. 启动 GUI")
        gui_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        gui_btn.setToolTip(gui_tip)
        gui_btn.clicked.connect(self._on_a2d_start_gui)
        cbtn_row.addWidget(gui_btn)
        pedal_btn = _eai_platform_bootstrap.QPushButton("3. ROS 服务 + 脚踏")
        pedal_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        pedal_btn.setToolTip("启动 robot-service / 手臂栈（含 pedal_controller）")
        pedal_btn.clicked.connect(self._on_a2d_ros_and_pedal)
        cbtn_row.addWidget(pedal_btn)
        cbtn_row.addStretch(1)
        calib_l.addLayout(cbtn_row)
        root.addWidget(calib_box)
        if not is_local:
            self.a2d_tracker_calib_btn = tracker_btn
            self.a2d_start_gui_btn = gui_btn
            self.a2d_ros_pedal_btn = pedal_btn

        # —— 查看标定 ——
        check_box = _eai_platform_bootstrap.QGroupBox("查看机器人标定")
        check_box.setStyleSheet(self._a2d_ops_group_style())
        check_l = _eai_platform_bootstrap.QVBoxLayout(check_box)
        check_l.setContentsMargins(8, 10, 8, 8)
        check_l.setSpacing(6)
        check_tip = _eai_platform_bootstrap.QLabel(_eai_configuration_policies.A2D_HMI_CALIB_TIP)
        check_tip.setWordWrap(True)
        check_tip.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        check_l.addWidget(check_tip)
        check_row = _eai_platform_bootstrap.QHBoxLayout()
        check_row.setSpacing(6)
        baseline_btn = _eai_platform_bootstrap.QPushButton("启动 baseline_check")
        baseline_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        baseline_btn.setToolTip(
            "ros2 launch reset_baseline baseline_check.launch.py（需已开 ROS 服务）"
        )
        baseline_btn.clicked.connect(self._on_a2d_baseline_check)
        check_row.addWidget(baseline_btn)
        hmi_btn = _eai_platform_bootstrap.QPushButton("HMI 操作说明")
        hmi_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        hmi_btn.clicked.connect(self._on_a2d_hmi_tip)
        check_row.addWidget(hmi_btn)
        check_row.addStretch(1)
        check_l.addLayout(check_row)
        root.addWidget(check_box)
        if not is_local:
            self.a2d_baseline_btn = baseline_btn
            self.a2d_hmi_tip_btn = hmi_btn

        # —— 模型 / HIL ——
        model_box = _eai_platform_bootstrap.QGroupBox("模型测试和 HIL")
        model_box.setStyleSheet(self._a2d_ops_group_style())
        model_l = _eai_platform_bootstrap.QVBoxLayout(model_box)
        model_l.setContentsMargins(8, 10, 8, 8)
        model_l.setSpacing(6)
        policy_row = _eai_platform_bootstrap.QHBoxLayout()
        policy_row.addWidget(_eai_platform_bootstrap.QLabel("psi-policy 根目录"))
        policy_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_policies.A2D_PSI_POLICY_ROOT_DEFAULT)
        policy_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        policy_edit.setToolTip(
            "含 .venv 与 model_inference_client_cena；可用 A2D_PSI_POLICY_ROOT 覆盖"
        )
        policy_row.addWidget(policy_edit, 1)
        policy_browse = _eai_platform_bootstrap.QPushButton("浏览…")
        policy_browse.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        policy_browse.clicked.connect(self._on_a2d_browse_policy_root)
        policy_row.addWidget(policy_browse)
        model_l.addLayout(policy_row)
        ckpt_row = _eai_platform_bootstrap.QHBoxLayout()
        ckpt_row.addWidget(_eai_platform_bootstrap.QLabel("ckpt_path"))
        ckpt_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_policies.A2D_CKPT_PATH_DEFAULT)
        ckpt_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        ckpt_edit.setToolTip(
            "写入环境变量 CKPT_PATH（若 client 脚本读取）；"
            "也可在 outputs/ 下选 last.ckpt"
        )
        ckpt_row.addWidget(ckpt_edit, 1)
        ckpt_browse = _eai_platform_bootstrap.QPushButton("浏览…")
        ckpt_browse.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        ckpt_browse.clicked.connect(self._on_a2d_browse_ckpt)
        ckpt_row.addWidget(ckpt_browse)
        model_l.addLayout(ckpt_row)
        mbtn_row = _eai_platform_bootstrap.QHBoxLayout()
        mbtn_row.setSpacing(6)
        infer_btn = _eai_platform_bootstrap.QPushButton("启动 inference server")
        infer_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        infer_btn.setToolTip(
            "ros2 run model_inference run_inference_server"
            + ("（本机）" if is_local else "（容器内 / docker exec）")
        )
        infer_btn.clicked.connect(self._on_a2d_infer_server)
        mbtn_row.addWidget(infer_btn)
        policy_infer_btn = _eai_platform_bootstrap.QPushButton("模型推理模式")
        policy_infer_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        policy_infer_btn.setToolTip("source .venv && python run_policy_inference.py")
        policy_infer_btn.clicked.connect(
            lambda: self._on_a2d_policy_client(hil=False)
        )
        mbtn_row.addWidget(policy_infer_btn)
        hil_btn = _eai_platform_bootstrap.QPushButton("HIL 模式")
        hil_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        hil_btn.setToolTip(
            "python run_policy_inference.py 2>&1 | tee hil_repro_act.log"
        )
        hil_btn.clicked.connect(lambda: self._on_a2d_policy_client(hil=True))
        mbtn_row.addWidget(hil_btn)
        outputs_btn = _eai_platform_bootstrap.QPushButton("打开 outputs")
        outputs_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        outputs_btn.clicked.connect(self._on_a2d_open_outputs)
        mbtn_row.addWidget(outputs_btn)
        mbtn_row.addStretch(1)
        model_l.addLayout(mbtn_row)
        root.addWidget(model_box)
        if is_local:
            self.a2d_local_policy_root_edit = policy_edit
            self.a2d_local_ckpt_edit = ckpt_edit
        else:
            self.a2d_policy_root_edit = policy_edit
            self.a2d_ckpt_edit = ckpt_edit
            self.a2d_policy_browse_btn = policy_browse
            self.a2d_ckpt_browse_btn = ckpt_browse
            self.a2d_infer_server_btn = infer_btn
            self.a2d_policy_infer_btn = policy_infer_btn
            self.a2d_hil_btn = hil_btn
            self.a2d_open_outputs_btn = outputs_btn

        # —— 常见问题 ——
        faq_box = _eai_platform_bootstrap.QGroupBox("常见问题和解决办法")
        faq_box.setStyleSheet(self._a2d_ops_group_style())
        faq_l = _eai_platform_bootstrap.QVBoxLayout(faq_box)
        faq_l.setContentsMargins(8, 10, 8, 8)
        faq_l.setSpacing(6)
        faq_hint = _eai_platform_bootstrap.QLabel(
            "Steam VR / GUI 打不开（刚开机或重启后）→ xhost +；"
            "tracker 连不上 → 重启 Steam VR；手部异常 → rui-hand 排查。"
        )
        faq_hint.setWordWrap(True)
        faq_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        faq_l.addWidget(faq_hint)
        fbtn_row = _eai_platform_bootstrap.QHBoxLayout()
        fbtn_row.setSpacing(6)
        xhost_btn = _eai_platform_bootstrap.QPushButton("xhost +")
        xhost_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        xhost_btn.setToolTip("宿主机: xhost +（GUI / Steam VR 打不开时）")
        xhost_btn.clicked.connect(self._on_a2d_xhost)
        fbtn_row.addWidget(xhost_btn)
        steamvr_btn = _eai_platform_bootstrap.QPushButton("重启 Steam VR")
        steamvr_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        steamvr_btn.setToolTip("ros2 launch vr_tracker steamvr.launch.py")
        steamvr_btn.clicked.connect(self._on_a2d_steamvr)
        fbtn_row.addWidget(steamvr_btn)
        rui_btn = _eai_platform_bootstrap.QPushButton("检查手 (rui-hand)")
        rui_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        rui_btn.setToolTip(
            "sudo rui-hand → 设备连接 → RS485 → 波特率 460800 → 串口扫描打开端口"
        )
        rui_btn.clicked.connect(self._on_a2d_rui_hand)
        fbtn_row.addWidget(rui_btn)
        fbtn_row.addStretch(1)
        faq_l.addLayout(fbtn_row)
        root.addWidget(faq_box)
        if not is_local:
            self.a2d_xhost_btn = xhost_btn
            self.a2d_steamvr_btn = steamvr_btn
            self.a2d_rui_hand_btn = rui_btn

        # —— 观看视频 ——
        video_box = _eai_platform_bootstrap.QGroupBox("观看采集的视频")
        video_box.setStyleSheet(self._a2d_ops_group_style())
        video_l = _eai_platform_bootstrap.QVBoxLayout(video_box)
        video_l.setContentsMargins(8, 10, 8, 8)
        video_l.setSpacing(6)
        rrd_row = _eai_platform_bootstrap.QHBoxLayout()
        rrd_row.addWidget(_eai_platform_bootstrap.QLabel("rrd_path"))
        rrd_edit = _eai_platform_bootstrap.QLineEdit(_eai_configuration_policies.A2D_RRD_PATH_DEFAULT)
        rrd_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        rrd_edit.setToolTip(
            "ros2 launch rrd_replay rrd_replay.launch.py rrd_path:=…"
        )
        rrd_row.addWidget(rrd_edit, 1)
        rrd_browse = _eai_platform_bootstrap.QPushButton("浏览…")
        rrd_browse.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        rrd_browse.clicked.connect(self._on_a2d_browse_rrd)
        rrd_row.addWidget(rrd_browse)
        video_l.addLayout(rrd_row)
        vbtn_row = _eai_platform_bootstrap.QHBoxLayout()
        vbtn_row.setSpacing(6)
        rrd_play = _eai_platform_bootstrap.QPushButton("播放 RRD")
        rrd_play.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        rrd_play.setToolTip("启动 rrd_replay 观看采集视频")
        rrd_play.clicked.connect(self._on_a2d_play_rrd)
        vbtn_row.addWidget(rrd_play)
        vbtn_row.addStretch(1)
        video_l.addLayout(vbtn_row)
        root.addWidget(video_box)
        if is_local:
            self.a2d_local_rrd_edit = rrd_edit
        else:
            self.a2d_rrd_edit = rrd_edit
            self.a2d_rrd_browse_btn = rrd_browse
            self.a2d_rrd_play_btn = rrd_play

        root.addStretch(1)
        return page

    def _a2d_ops_mode(self) -> str:
        tabs = getattr(self, "a2d_ops_mode_tabs", None)
        if tabs is not None and tabs.currentIndex() == 1:
            return "local"
        return "docker"

    def _a2d_ops_is_local(self) -> bool:
        return self._a2d_ops_mode() == "local"

    def _a2d_policy_root_edit_widget(self) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QLineEdit]:
        if self._a2d_ops_is_local():
            return getattr(self, "a2d_local_policy_root_edit", None)
        return getattr(self, "a2d_policy_root_edit", None)

    def _a2d_ckpt_edit_widget(self) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QLineEdit]:
        if self._a2d_ops_is_local():
            return getattr(self, "a2d_local_ckpt_edit", None)
        return getattr(self, "a2d_ckpt_edit", None)

    def _a2d_rrd_edit_widget(self) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QLineEdit]:
        if self._a2d_ops_is_local():
            return getattr(self, "a2d_local_rrd_edit", None)
        return getattr(self, "a2d_rrd_edit", None)

    def _a2d_local_workspace(self) -> str:
        edit = getattr(self, "a2d_local_ws_edit", None)
        if edit is not None:
            raw = _eai_platform_bootstrap.os.path.expanduser(edit.text().strip())
            if raw:
                return raw
        return _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()

    def _a2d_launch_ros(self, ros_cmd: str, **kwargs) -> _eai_platform_bootstrap.Tuple[bool, str]:
        force_local = self._a2d_ops_is_local()
        container = "" if force_local else self._a2d_container_name()
        if force_local and "cwd" not in kwargs:
            kwargs["cwd"] = self._a2d_local_workspace()
        ok, msg = _eai_adapters_robot_services.launch_a2d_ros_detached(
            ros_cmd,
            force_local=force_local,
            container=container,
            **kwargs,
        )
        if not ok and force_local and (
            "不存在" in msg or "缺少 a2d-tele" in msg or "install" in msg
        ):
            _eai_platform_bootstrap.QMessageBox.warning(self, "A2D local 启动失败", msg)
        return ok, msg

    def _a2d_container_name(self) -> str:
        edit = getattr(self, "a2d_docker_name_edit", None)
        raw = edit.text().strip() if edit is not None else ""
        return _eai_adapters_robot_services.a2d_docker_container_name(raw)

    def _log_a2d_ops(self, line: str) -> None:
        mode = self._a2d_ops_mode()
        self._append_real_eval_log(f"[A2D运维/{mode}] {line}")
        if hasattr(self, "status_bar") and self.status_bar is not None:
            self.status_bar.showMessage(line[:120], 5000)

    def _on_a2d_open_doc(self) -> None:
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl(_eai_configuration_policies.A2D_OPS_DOC_URL))
        self._log_a2d_ops(f"打开文档: {_eai_configuration_policies.A2D_OPS_DOC_URL}")

    def _on_a2d_local_check_env(self) -> None:
        edit = getattr(self, "a2d_local_ws_edit", None)
        ws = ""
        if edit is not None:
            ws = _eai_platform_bootstrap.os.path.expanduser(edit.text().strip())
        if not ws:
            ws = _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()
        setup = _eai_platform_bootstrap.os.path.join(ws, "install", "setup.bash")
        scripts = _eai_adapters_robot_launch_utils.resolve_scripts_pack_dir()
        lines = [
            f"workspace={ws}",
            f"install/setup.bash={'OK' if _eai_platform_bootstrap.os.path.isfile(setup) else 'MISSING'}",
            f"scripts_pack={scripts or 'MISSING'}",
            f"ROS_DISTRO={_eai_platform_bootstrap.os.environ.get('ROS_DISTRO', '?')}",
            f"RMW={_eai_platform_bootstrap.os.environ.get('RMW_IMPLEMENTATION', '?')}",
            f"DOMAIN={_eai_platform_bootstrap.os.environ.get('ROS_DOMAIN_ID', '?')}",
        ]
        ok = _eai_platform_bootstrap.os.path.isfile(setup)
        status = getattr(self, "a2d_local_env_status", None)
        text = ("就绪 · " if ok else "未就绪 · ") + " | ".join(lines)
        if status is not None:
            status.setText(text)
            status.setStyleSheet(
                f"color: {'#6aab73' if ok else '#c97b7b'};"
            )
        self._log_a2d_ops(("环境OK\n" if ok else "环境缺失\n") + "\n".join(lines))

    def _on_a2d_local_open_ws(self) -> None:
        edit = getattr(self, "a2d_local_ws_edit", None)
        ws = ""
        if edit is not None:
            ws = _eai_platform_bootstrap.os.path.expanduser(edit.text().strip())
        if not ws:
            ws = _eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir()
        if not _eai_platform_bootstrap.os.path.isdir(ws):
            self._log_a2d_ops(f"workspace 不存在: {ws}")
            _eai_platform_bootstrap.QMessageBox.warning(self, "本机环境", f"目录不存在:\n{ws}")
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(_eai_platform_bootstrap.os.path.abspath(ws)))
        self._log_a2d_ops(f"打开 workspace: {ws}")

    def _on_a2d_local_refresh_ws(self) -> None:
        edit = getattr(self, "a2d_local_ws_edit", None)
        if edit is None:
            return
        edit.setText(_eai_adapters_robot_launch_utils.resolve_a2d_workspace_dir())
        self._on_a2d_local_check_env()

    def _on_a2d_docker_ps(self) -> None:
        ok, text = _eai_adapters_robot_services.run_a2d_docker_action("ps", self._a2d_container_name(), capture=True)
        self._log_a2d_ops(("OK\n" if ok else "FAIL\n") + text)

    def _on_a2d_docker_stop(self) -> None:
        name = self._a2d_container_name()
        reply = _eai_platform_bootstrap.QMessageBox.question(
            self,
            "关闭容器",
            f"确认 docker stop {name}？\n"
            "若当前 viewer 跑在该容器内，界面会一起退出。",
            _eai_platform_bootstrap.QMessageBox.Yes | _eai_platform_bootstrap.QMessageBox.No,
            _eai_platform_bootstrap.QMessageBox.No,
        )
        if reply != _eai_platform_bootstrap.QMessageBox.Yes:
            return
        ok, text = _eai_adapters_robot_services.run_a2d_docker_action("stop", name)
        self._log_a2d_ops(text if ok else f"FAIL: {text}")

    def _on_a2d_docker_start(self) -> None:
        ok, text = _eai_adapters_robot_services.run_a2d_docker_action("start", self._a2d_container_name())
        self._log_a2d_ops(text if ok else f"FAIL: {text}")

    def _on_a2d_docker_restart(self) -> None:
        name = self._a2d_container_name()
        reply = _eai_platform_bootstrap.QMessageBox.question(
            self,
            "重启容器",
            f"确认 docker restart {name}？\n"
            "若当前 viewer 跑在该容器内，界面会短暂中断。",
            _eai_platform_bootstrap.QMessageBox.Yes | _eai_platform_bootstrap.QMessageBox.No,
            _eai_platform_bootstrap.QMessageBox.No,
        )
        if reply != _eai_platform_bootstrap.QMessageBox.Yes:
            return
        ok, text = _eai_adapters_robot_services.run_a2d_docker_action("restart", name)
        self._log_a2d_ops(text if ok else f"FAIL: {text}")

    def _on_a2d_tracker_calib(self) -> None:
        ok, msg = self._a2d_launch_ros("ros2 run vr_calibration calibrate_node")
        self._log_a2d_ops(msg if ok else f"FAIL: {msg}")

    def _on_a2d_start_gui(self) -> None:
        ok, msg = self._a2d_launch_ros(
            "ros2 launch scripts_pack start_gui.launch.py"
        )
        self._log_a2d_ops(msg if ok else f"FAIL: {msg}")

    def _on_a2d_ros_and_pedal(self) -> None:
        mode = self._a2d_ops_mode()
        self._log_a2d_ops(f"启动 ROS 服务与脚踏（A2D/{mode}）…")
        self._append_real_eval_log(
            f"[栈] 后端=A2D/{mode}，采集标定流程 → RobotStackLauncher"
        )
        self._on_robot_stack_clicked()

    def _on_a2d_baseline_check(self) -> None:
        ok, msg = self._a2d_launch_ros(
            "ros2 launch reset_baseline baseline_check.launch.py"
        )
        self._log_a2d_ops(msg if ok else f"FAIL: {msg}")
        if ok:
            _eai_platform_bootstrap.QMessageBox.information(self, "查看机器人标定", _eai_configuration_policies.A2D_HMI_CALIB_TIP)

    def _on_a2d_hmi_tip(self) -> None:
        _eai_platform_bootstrap.QMessageBox.information(self, "HMI 标定检查", _eai_configuration_policies.A2D_HMI_CALIB_TIP)

    def _on_a2d_browse_policy_root(self) -> None:
        edit = self._a2d_policy_root_edit_widget()
        if edit is None:
            return
        cur = edit.text().strip() or _eai_configuration_policies.A2D_PSI_POLICY_ROOT_DEFAULT
        selected = _eai_platform_bootstrap.QFileDialog.getExistingDirectory(
            self, "选择 psi-policy-v2 根目录", cur
        )
        if selected:
            edit.setText(selected)

    def _on_a2d_browse_ckpt(self) -> None:
        edit = self._a2d_ckpt_edit_widget()
        if edit is None:
            return
        cur = edit.text().strip() or _eai_configuration_policies.A2D_CKPT_PATH_DEFAULT
        start = _eai_platform_bootstrap.os.path.dirname(cur) if cur else _eai_configuration_policies.A2D_PSI_POLICY_ROOT_DEFAULT
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self,
            "选择 checkpoint",
            start,
            "Checkpoint (*.ckpt *.pt *.pth);;All (*)",
        )
        if path:
            edit.setText(path)

    def _on_a2d_infer_server(self) -> None:
        ok, msg = self._a2d_launch_ros(
            "ros2 run model_inference run_inference_server"
        )
        self._log_a2d_ops(msg if ok else f"FAIL: {msg}")

    def _on_a2d_policy_client(self, *, hil: bool) -> None:
        policy_edit = self._a2d_policy_root_edit_widget()
        ckpt_edit = self._a2d_ckpt_edit_widget()
        if policy_edit is None:
            return
        root = _eai_platform_bootstrap.os.path.expanduser(policy_edit.text().strip())
        client_dir = _eai_platform_bootstrap.os.path.join(root, "model_inference_client_cena")
        venv_act = _eai_platform_bootstrap.os.path.join(root, ".venv", "bin", "activate")
        if not _eai_platform_bootstrap.os.path.isdir(client_dir):
            self._log_a2d_ops(f"未找到 client 目录: {client_dir}")
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "模型测试 / HIL",
                f"未找到目录:\n{client_dir}\n\n请确认 psi-policy 根目录。",
            )
            return
        if not _eai_platform_bootstrap.os.path.isfile(venv_act):
            self._log_a2d_ops(f"未找到 venv: {venv_act}（需出现 psipolicy-env）")
            _eai_platform_bootstrap.QMessageBox.warning(
                self,
                "模型测试 / HIL",
                f"未找到 .venv:\n{venv_act}\n\n"
                "请先在该目录创建并激活虚拟环境。",
            )
            return
        ckpt = ckpt_edit.text().strip() if ckpt_edit is not None else ""
        env_bits = []
        if ckpt:
            env_bits.append(f"export CKPT_PATH={_eai_platform_bootstrap.shlex.quote(ckpt)}")
            env_bits.append(f"export ckpt_path={_eai_platform_bootstrap.shlex.quote(ckpt)}")
        env_export = " && ".join(env_bits)
        env_prefix = f"{env_export} && " if env_export else ""
        if hil:
            py_cmd = "python run_policy_inference.py 2>&1 | tee hil_repro_act.log"
            label = "HIL"
        else:
            py_cmd = "python run_policy_inference.py"
            label = "模型推理"
        full = (
            f"set -e && cd {_eai_platform_bootstrap.shlex.quote(root)} && source .venv/bin/activate && "
            f"cd {_eai_platform_bootstrap.shlex.quote(client_dir)} && {env_prefix}{py_cmd}"
        )
        if _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", full], client_dir):
            self._log_a2d_ops(f"已启动{label}: {py_cmd} @ {client_dir}")
        else:
            self._log_a2d_ops(f"启动{label}失败")

    def _on_a2d_open_outputs(self) -> None:
        policy_edit = self._a2d_policy_root_edit_widget()
        if policy_edit is None:
            return
        root = _eai_platform_bootstrap.os.path.expanduser(policy_edit.text().strip())
        out_dir = _eai_platform_bootstrap.os.path.join(root, "outputs")
        if not _eai_platform_bootstrap.os.path.isdir(out_dir):
            self._log_a2d_ops(f"outputs 不存在: {out_dir}")
            _eai_platform_bootstrap.QMessageBox.information(self, "outputs", f"目录不存在:\n{out_dir}")
            return
        _eai_platform_bootstrap.QDesktopServices.openUrl(_eai_platform_bootstrap.QUrl.fromLocalFile(_eai_platform_bootstrap.os.path.abspath(out_dir)))
        self._log_a2d_ops(f"打开 outputs: {out_dir}")

    def _on_a2d_xhost(self) -> None:
        try:
            result = _eai_platform_bootstrap.subprocess.run(
                ["bash", "-lc", "xhost +"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            text = (result.stdout or result.stderr or "").strip() or f"exit={result.returncode}"
            self._log_a2d_ops(
                f"xhost +: {text}" if result.returncode == 0 else f"xhost + FAIL: {text}"
            )
        except Exception as exc:
            self._log_a2d_ops(f"xhost + 异常: {exc}")

    def _on_a2d_steamvr(self) -> None:
        ok, msg = self._a2d_launch_ros(
            "ros2 launch vr_tracker steamvr.launch.py"
        )
        self._log_a2d_ops(msg if ok else f"FAIL: {msg}")

    def _on_a2d_rui_hand(self) -> None:
        _eai_platform_bootstrap.QMessageBox.information(
            self,
            "检查手 (rui-hand)",
            "将尝试启动 sudo rui-hand。\n\n"
            "GUI 内操作：点「设备连接」→ RS485 → 波特率 460800 → "
            "串口扫描 → 打开端口。",
        )
        if _eai_platform_bootstrap.QProcess.startDetached("bash", ["-lc", "sudo rui-hand"]):
            self._log_a2d_ops("已启动 sudo rui-hand")
        else:
            self._log_a2d_ops("启动 rui-hand 失败（可能需要在终端手动 sudo）")

    def _on_a2d_browse_rrd(self) -> None:
        edit = self._a2d_rrd_edit_widget()
        if edit is None:
            return
        cur = edit.text().strip() or _eai_configuration_policies.A2D_RRD_PATH_DEFAULT
        start = _eai_platform_bootstrap.os.path.dirname(cur) if cur else "/root/project/rerun_data"
        path, _ = _eai_platform_bootstrap.QFileDialog.getOpenFileName(
            self, "选择采集 RRD", start, "Rerun (*.rrd);;All (*)"
        )
        if path:
            edit.setText(path)

    def _on_a2d_play_rrd(self) -> None:
        edit = self._a2d_rrd_edit_widget()
        path = edit.text().strip() if edit is not None else ""
        if not path:
            _eai_platform_bootstrap.QMessageBox.warning(self, "观看采集视频", "请填写 rrd_path")
            return
        # 优先走已有回放启动器（与「回放」页一致）
        if hasattr(self, "_start_rrd_replay"):
            ok = self._start_rrd_replay(path, 1)
            if ok:
                self._log_a2d_ops(f"已启动 RRD 回放: {path}")
                return
            self._log_a2d_ops(f"内置回放启动未成功，改用 launch: {path}")
        quoted = path.replace("'", "'\\''")
        ok, msg = self._a2d_launch_ros(
            f"ros2 launch rrd_replay rrd_replay.launch.py rrd_path:='{quoted}'"
        )
        self._log_a2d_ops(msg if ok else f"FAIL: {msg}")

    def _close_real_eval_local_cam(self) -> None:
        cap = self._real_eval_local_cap
        self._real_eval_local_cap = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    def _stop_real_eval_local_cam(self) -> None:
        self._real_eval_local_cam_active = False
        self._real_eval_local_cam_busy = False
        if hasattr(self, "_real_eval_local_cam_timer"):
            self._real_eval_local_cam_timer.stop()
        self._close_real_eval_local_cam()
        if hasattr(self, "real_eval_cam_preview_btn"):
            self.real_eval_cam_preview_btn.setText("本地摄像头预览")
        if hasattr(self, "real_eval_cam_status_label"):
            self.real_eval_cam_status_label.setText("本地摄像头: 未开启")
            self.real_eval_cam_status_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        if hasattr(self, "real_eval_cam_preview_label"):
            self.real_eval_cam_preview_label.setPixmap(_eai_platform_bootstrap.QPixmap())
            self.real_eval_cam_preview_label.setText(
                "点击「本地摄像头预览」显示本机画面（Remote SSH/VNC 需先推流，点「用法」）"
            )
        if hasattr(self, "real_eval_cam_url_edit"):
            self.real_eval_cam_url_edit.setEnabled(True)

    def _open_real_eval_local_cam_source(self, url_or_dev: str) -> _eai_platform_bootstrap.Tuple[bool, str]:
        """打开网络流 URL，或本机 /dev/video* / 数字 index。"""
        src = (url_or_dev or "").strip()
        if not src:
            return False, "请填写网络流 URL（例如 http://127.0.0.1:8090/cam.mjpg）"
        self._close_real_eval_local_cam()
        cap = None
        try:
            is_url = "://" in src or src.lower().endswith((".mjpg", ".mjpeg"))
            if is_url:
                cap = _eai_platform_bootstrap.cv2.VideoCapture(src, _eai_platform_bootstrap.cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = _eai_platform_bootstrap.cv2.VideoCapture(src)
            else:
                # 本机设备：/dev/videoN 或纯数字 index
                index: object = src
                if src.isdigit():
                    index = int(src)
                elif src.startswith("/dev/video") and src[len("/dev/video") :].isdigit():
                    index = int(src[len("/dev/video") :])
                if isinstance(index, int):
                    cap = _eai_platform_bootstrap.cv2.VideoCapture(index, _eai_platform_bootstrap.cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap.release()
                        cap = _eai_platform_bootstrap.cv2.VideoCapture(index)
                else:
                    cap = _eai_platform_bootstrap.cv2.VideoCapture(src)
            if not cap.isOpened():
                hint = (
                    f"无法打开:\n{src}\n\n"
                    "Remote SSH / VNC 下请先在 Mac 本机推流并建立 SSH -R 隧道（点「用法」）。"
                )
                return False, hint
            try:
                cap.set(_eai_platform_bootstrap.cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            deadline = _eai_platform_bootstrap.time.time() + (8.0 if is_url else 2.0)
            frame = None
            ok = False
            while _eai_platform_bootstrap.time.time() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    break
                _eai_platform_bootstrap.time.sleep(0.05)
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                cap.release()
                return False, f"已连接但读不到画面:\n{src}"
            self._real_eval_local_cap = cap
            return True, src
        except Exception as exc:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return False, f"打开摄像头失败: {exc}"

    def _append_local_cam_log(self, line: str) -> None:
        if not hasattr(self, "local_cam_log_edit"):
            return
        self.local_cam_log_edit.append(line)
        scrollbar = self.local_cam_log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _refresh_local_cam_device_list(self, select: str = "") -> None:
        if not hasattr(self, "local_cam_device_combo"):
            return
        want = (select or self.local_cam_device_edit.text().strip() or "").strip()
        if want.startswith("/dev/video") and want[len("/dev/video") :].isdigit():
            want = want[len("/dev/video") :]
        devices = _eai_perception_image_geometry.list_local_v4l_devices()
        self._local_cam_device_combo_syncing = True
        self.local_cam_device_combo.clear()
        if not devices:
            self.local_cam_device_combo.addItem("未发现 /dev/video*（macOS 请填 0）", "0")
        else:
            for idx, label in devices:
                self.local_cam_device_combo.addItem(label, idx)
        pick = -1
        if want:
            for i in range(self.local_cam_device_combo.count()):
                if str(self.local_cam_device_combo.itemData(i)) == want:
                    pick = i
                    break
        if pick < 0 and devices:
            sug = _eai_perception_image_geometry.suggest_local_webcam_device()
            for i in range(self.local_cam_device_combo.count()):
                if str(self.local_cam_device_combo.itemData(i)) == sug:
                    pick = i
                    break
        if pick < 0:
            pick = 0
        self.local_cam_device_combo.setCurrentIndex(pick)
        data = self.local_cam_device_combo.currentData()
        if data is not None and hasattr(self, "local_cam_device_edit"):
            self.local_cam_device_edit.setText(str(data))
        self._local_cam_device_combo_syncing = False
        n = len(devices)
        self._append_local_cam_log(
            f"[设备] 扫描到 {n} 个采集节点"
            + (f"，当前 {self.local_cam_device_edit.text().strip()}" if n else "")
        )

    def _set_local_cam_device_controls_enabled(self, enabled: bool) -> None:
        if hasattr(self, "local_cam_device_edit"):
            self.local_cam_device_edit.setEnabled(enabled)
        if hasattr(self, "local_cam_device_combo"):
            self.local_cam_device_combo.setEnabled(enabled)
        if hasattr(self, "local_cam_refresh_dev_btn"):
            # 推流占用设备时仍允许刷新列表查看，但监控中禁用改设备
            if getattr(self, "_local_cam_preview_mode", "") == "device":
                self.local_cam_refresh_dev_btn.setEnabled(enabled)
            else:
                self.local_cam_refresh_dev_btn.setEnabled(True)

    def _confirm_local_cam_on_remote_host(self, action: str) -> bool:
        """SSH 远端上点推流/隧道时提示；返回 False 表示取消。"""
        if not (_eai_platform_bootstrap.os.environ.get("SSH_CONNECTION") or _eai_platform_bootstrap.os.environ.get("SSH_CLIENT")):
            return True
        reply = _eai_platform_bootstrap.QMessageBox.question(
            self,
            "本地摄像头",
            (
                "检测到界面运行在 SSH 远端。\n"
                f"「{action}」应在有摄像头的本机执行。\n\n"
                "若摄像头已插在当前机器（如 RealSense），可选「是」继续。\n"
                "若摄像头在 Mac，建议在 Mac 运行:\n"
                "  python3 examples/local_webcam_forward_ui.py\n"
                "（可用本页「复制 Mac 助手命令」）\n\n"
                "仍要在当前机器尝试吗？"
            ),
            _eai_platform_bootstrap.QMessageBox.Yes | _eai_platform_bootstrap.QMessageBox.No,
            _eai_platform_bootstrap.QMessageBox.No,
        )
        return reply == _eai_platform_bootstrap.QMessageBox.Yes

    def _close_local_cam_preview(self) -> None:
        cap = self._local_cam_preview_cap
        self._local_cam_preview_cap = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    def _stop_local_cam_preview(self) -> None:
        was_mode = getattr(self, "_local_cam_preview_mode", "")
        self._local_cam_preview_active = False
        self._local_cam_preview_busy = False
        self._local_cam_preview_mode = ""
        if hasattr(self, "_local_cam_preview_timer"):
            self._local_cam_preview_timer.stop()
        self._close_local_cam_preview()
        if hasattr(self, "local_cam_preview_btn"):
            self.local_cam_preview_btn.setText("打开预览")
        if hasattr(self, "local_cam_monitor_btn"):
            self.local_cam_monitor_btn.setText("打开本机监控")
        if hasattr(self, "local_cam_preview_status"):
            self.local_cam_preview_status.setText("预览: 未开启")
            self.local_cam_preview_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        if hasattr(self, "local_cam_monitor_status"):
            self.local_cam_monitor_status.setText("监控: 未开启")
            self.local_cam_monitor_status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        if hasattr(self, "local_cam_preview_label"):
            self.local_cam_preview_label.setPixmap(_eai_platform_bootstrap.QPixmap())
            self.local_cam_preview_label.setText(
                "点「打开本机监控」直接看本机摄像头，或推流后点「打开预览」"
            )
        if hasattr(self, "local_cam_url_edit"):
            self.local_cam_url_edit.setEnabled(True)
        mjpeg_running = False
        launcher = getattr(self, "_local_cam_mjpeg_launcher", None)
        if launcher is not None:
            mjpeg_running = launcher.is_running()
        if was_mode == "device" and not mjpeg_running:
            self._set_local_cam_device_controls_enabled(True)

