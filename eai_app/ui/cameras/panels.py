"""ui.cameras.panels — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_adapters_ros_bridges = module_ref('eai_app.adapters.ros.bridges')
_eai_configuration_perception = module_ref('eai_app.configuration.perception')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_image_geometry = module_ref('eai_app.perception.image_geometry')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_images = module_ref('eai_app.ui.widgets.images')

class CameraPanel(_eai_platform_bootstrap.QWidget):
    def __init__(
        self,
        topic: str,
        get_paired_depth: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]]] = None,
        get_intrinsics_for_depth: _eai_platform_bootstrap.Optional[
            _eai_platform_bootstrap.Callable[[str, int, int], _eai_platform_bootstrap.Tuple[float, float, float, float]]
        ] = None,
        get_depth_topic: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], _eai_platform_bootstrap.Optional[str]]] = None,
        get_camera_frame_id: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], str]] = None,
        resolve_segment_camera_frame: _eai_platform_bootstrap.Optional[
            _eai_platform_bootstrap.Callable[[str, _eai_platform_bootstrap.Optional[str]], str]
        ] = None,
        is_paired_depth_enabled: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], bool]] = None,
        on_segment_pose: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[_eai_perception_pose_estimation.SegmentPoseTarget], None]] = None,
        on_click_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str, int, int], None]] = None,
        status_callback: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], None]] = None,
        parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.topic = topic
        self._get_paired_depth = get_paired_depth
        self._get_intrinsics_for_depth = get_intrinsics_for_depth
        self._get_depth_topic = get_depth_topic
        self._get_camera_frame_id = get_camera_frame_id
        self._resolve_segment_camera_frame = resolve_segment_camera_frame
        self._is_paired_depth_enabled = is_paired_depth_enabled
        self._on_segment_pose = on_segment_pose
        self._on_click_uv = on_click_uv
        self._status_callback = status_callback
        self._frame_count = 0
        self._last_fps_time = _eai_platform_bootstrap.time.time()
        self._fps = 0.0
        self._latest_image: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None

        self.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
        layout = _eai_platform_bootstrap.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        title = topic.split("/")[-1] or topic
        self.title_label = _eai_platform_bootstrap.QLabel(title)
        self.title_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_TITLE, _eai_platform_bootstrap.QFont.Bold))
        self.title_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.title_label.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Fixed)
        layout.addWidget(self.title_label)

        self.image_label = _eai_ui_widgets_images.ClickableImageLabel("等待图像...（点击选点，再按「调用分割」确认）")
        self.image_label.clicked_pixel.connect(self._on_pixel_clicked)
        layout.addWidget(self.image_label, stretch=1)

        self.info_label = _eai_platform_bootstrap.QLabel(topic)
        self.info_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        self.info_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.info_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        self.info_label.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Fixed)
        layout.addWidget(self.info_label)

        self.coord_label = _eai_platform_bootstrap.QLabel("点击图像选择提示点；再点「调用分割 / SAM3 / FP」执行")
        self.coord_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        self.coord_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.coord_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE_BRIGHT};")
        self.coord_label.setWordWrap(True)
        self.coord_label.setMaximumHeight(40)
        self.coord_label.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Fixed)
        layout.addWidget(self.coord_label)

        self._pose_bridge = _eai_adapters_ros_bridges.PoseComputeBridge()
        self._pose_bridge.finished.connect(self._on_pose_compute_finished)
        self._pose_busy = False
        self._pending_seed_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]] = None

    def _show_seed_marker(self, u: int, v: int, hint: str) -> None:
        self._pending_seed_uv = (u, v)
        self.image_label.set_segment_overlay(None, seed_uv=(float(u), float(v)))
        self.coord_label.setText(hint)

    def invoke_segment_at_seed(
        self,
        u: _eai_platform_bootstrap.Optional[int] = None,
        v: _eai_platform_bootstrap.Optional[int] = None,
    ) -> str:
        """按提示点启动立体分割+位姿（由按钮触发，不自动在点击时执行）。"""
        if self._pose_busy:
            return f"[{self.topic}] 分割进行中，请稍候"
        if self._latest_image is None:
            return f"[{self.topic}] 尚无彩色图像"
        if u is None or v is None:
            if self._pending_seed_uv is None:
                return f"[{self.topic}] 请先点击图像选择提示点"
            u, v = self._pending_seed_uv
        h, w = self._latest_image.shape[:2]
        u = max(0, min(w - 1, int(u)))
        v = max(0, min(h - 1, int(v)))
        self._pending_seed_uv = (u, v)
        self.image_label.set_segment_overlay(None, seed_uv=(float(u), float(v)))

        if self._is_paired_depth_enabled is not None and not self._is_paired_depth_enabled(
            self.topic
        ):
            return f"[{self.topic}] 未勾选配对 depth，无法分割"
        depth = self._get_paired_depth(self.topic) if self._get_paired_depth else None
        depth_topic = self._get_depth_topic(self.topic) if self._get_depth_topic else None
        if depth is None or not depth_topic or self._get_intrinsics_for_depth is None:
            return f"[{self.topic}] 无 depth 数据，无法分割"
        dh, dw = depth.shape[:2]
        fx, fy, cx, cy = self._get_intrinsics_for_depth(depth_topic, dw, dh)
        u_d, v_d = _eai_perception_image_geometry.scale_uv_to_shape(u, v, self._latest_image.shape[:2], (dh, dw))
        self._start_pose_compute(
            depth, self._latest_image, u_d, v_d, fx, fy, cx, cy, display_u=u
        )
        return f"[{self.topic}] 已开始分割 @ ({u}, {v})"

    def _start_pose_compute(
        self,
        depth: _eai_platform_bootstrap.np.ndarray,
        color: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        u: int,
        v: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        display_u: int,
    ) -> None:
        if self._pose_busy:
            return
        self._pose_busy = True
        self.coord_label.setText("立体分割计算中...")

        def _work() -> None:
            overlay = None
            try:
                depth_copy = depth.copy()
                color_copy = color.copy() if color is not None else None
                result, info = _eai_perception_pose_estimation.run_stereo_pose_pipeline_resampled(
                    depth_copy, u, v, fx, fy, cx, cy, color_bgr=color_copy
                )
                if result is not None and color_copy is not None:
                    overlay = _eai_perception_image_geometry.build_pose_overlay_image(
                        color_copy, result, (fx, fy, cx, cy)
                    )
            except Exception as exc:
                result, info, overlay = None, f"点击 ({display_u}, {v})  |  计算错误: {exc}", None
            self._pose_bridge.finished.emit(result, info, display_u, v, overlay)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_pose_compute_finished(
        self,
        result: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult],
        info: str,
        display_u: int,
        display_v: int,
        overlay: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    ) -> None:
        self._pose_busy = False
        if result is None:
            self.image_label.clear_segment_overlay()
        else:
            # 用 mask 叠加（随视频流持续重绘），避免 set_precomposed 下一帧被冲掉
            display_mask = _eai_perception_image_geometry.resize_mask_to_shape(result.mask, self._latest_image.shape[:2])
            dh, dw = result.mask.shape[:2]
            ch, cw = self._latest_image.shape[:2]
            cu, cv_pt = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(result.centroid_uv[0])),
                int(round(result.centroid_uv[1])),
                (dh, dw),
                (ch, cw),
            )
            contact_u, contact_v = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(result.contact_uv[0])),
                int(round(result.contact_uv[1])),
                (dh, dw),
                (ch, cw),
            )
            depth_topic = self._get_depth_topic(self.topic) if self._get_depth_topic else None
            intr = None
            if depth_topic and self._get_intrinsics_for_depth:
                fx, fy, cx, cy = self._get_intrinsics_for_depth(depth_topic, dw, dh)
                # 内参按 color 尺寸缩放，便于画 OBB
                sx = float(cw) / float(max(1, dw))
                sy = float(ch) / float(max(1, dh))
                intr = (fx * sx, fy * sy, cx * sx, cy * sy)
            self.image_label.set_segment_overlay(
                display_mask,
                (float(cu), float(cv_pt)),
                obb_corners=result.obb_corners,
                intrinsics=intr,
                contact_uv=(float(contact_u), float(contact_v)),
                seed_uv=(float(display_u), float(display_v)),
                pose_position=_eai_platform_bootstrap.np.asarray(result.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
                pose_rotation=_eai_platform_bootstrap.np.asarray(result.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
                pose_axis_len=float(max(result.obb_extents) * 0.55),
            )
            self._emit_segment_pose(result)
        self.coord_label.setText(info)
        if self._status_callback:
            self._status_callback(f"[{self.topic}] {info.splitlines()[0]}")

    def _emit_segment_pose(self, result: _eai_perception_pose_estimation.Object6DPoseResult) -> None:
        if self._on_segment_pose is None:
            return
        depth_topic = self._get_depth_topic(self.topic) if self._get_depth_topic else None
        if self._resolve_segment_camera_frame is not None:
            camera_frame = self._resolve_segment_camera_frame(self.topic, depth_topic)
        elif self._get_camera_frame_id is not None:
            camera_frame = self._get_camera_frame_id(depth_topic or self.topic)
        else:
            return
        if not camera_frame:
            hint = (
                f"[{self.topic}] 分割成功但缺少相机 frame_id，"
                "无法用于左臂移动（请确认 depth/color 已发布 header.frame_id）"
            )
            if self._status_callback:
                self._status_callback(hint)
            return
        target = _eai_perception_pose_estimation.SegmentPoseTarget.from_pose_result(result, camera_frame, self.topic)
        _eai_platform_bootstrap.QTimer.singleShot(0, lambda t=target: self._on_segment_pose(t))

    def show_sam3_mask(self, mask: _eai_platform_bootstrap.np.ndarray, seed_u: int, seed_v: int) -> None:
        if self._latest_image is None or not _eai_platform_bootstrap.np.asarray(mask).any():
            self.image_label.clear_segment_overlay()
            return
        display_mask = _eai_perception_image_geometry.resize_mask_to_shape(mask, self._latest_image.shape[:2])
        ys, xs = _eai_platform_bootstrap.np.where(display_mask)
        if len(xs) > 0:
            cu, cv = float(_eai_platform_bootstrap.np.mean(xs)), float(_eai_platform_bootstrap.np.mean(ys))
        else:
            cu, cv = float(seed_u), float(seed_v)
        # 随视频流持续重绘（绿填充 + 黄轮廓 + 红星提示点）
        self.image_label.set_segment_overlay(
            display_mask,
            (cu, cv),
            seed_uv=(float(seed_u), float(seed_v)),
        )
        self.coord_label.setText(
            f"SAM3 {int(display_mask.sum())} px @ ({seed_u}, {seed_v})  "
            f"质心 ({cu:.0f}, {cv:.0f})"
        )

    def _on_pixel_clicked(self, u: int, v: int) -> None:
        if self._on_click_uv is not None:
            self._on_click_uv(self.topic, u, v)
        if self._latest_image is None:
            return
        info = _eai_perception_image_geometry.format_color_pixel_info(u, v, self._latest_image)
        hint = f"{info}  |  已选提示点，请点「调用分割 / SAM3 / FP」确认"
        self._show_seed_marker(u, v, hint)
        if self._status_callback:
            self._status_callback(f"[{self.topic}] 已选提示点 ({u}, {v})")

    def update_frame(self, cv_image: _eai_platform_bootstrap.np.ndarray) -> None:
        # 不在此再 copy：帧已由 take_pending_ui_frames 交接给 UI 线程
        self._latest_image = _eai_platform_bootstrap.np.asarray(cv_image)
        self.image_label.set_source_image(self._latest_image)

        self._frame_count += 1
        now = _eai_platform_bootstrap.time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        h, w = cv_image.shape[:2]
        self.info_label.setText(f"{self.topic}  |  {w}x{h}  |  {self._fps:.1f} Hz")


class DepthPanel3D(_eai_platform_bootstrap.QWidget):
    """深度图面板：默认 2D 伪彩；3D 点云按需启用（全局仅允许一个 GL 上下文）。"""

    # 多路 depth 同时建 GLViewWidget 会触发 OpenGL context 丢失 / SIGSEGV
    _active_gl_panel: _eai_platform_bootstrap.Optional["DepthPanel3D"] = None

    def __init__(
        self,
        topic: str,
        get_intrinsics: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str, int, int], _eai_platform_bootstrap.Tuple[float, float, float, float]]] = None,
        get_paired_color: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]]] = None,
        get_robot_state: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[], _eai_adapters_robot_geometry.RobotStateSnapshot]] = None,
        get_camera_frame_id: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], str]] = None,
        resolve_segment_camera_frame: _eai_platform_bootstrap.Optional[
            _eai_platform_bootstrap.Callable[[str, _eai_platform_bootstrap.Optional[str]], str]
        ] = None,
        get_tf_buffer: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Buffer]]] = None,
        on_segment_pose: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[_eai_perception_pose_estimation.SegmentPoseTarget], None]] = None,
        on_click_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str, int, int], None]] = None,
        status_callback: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], None]] = None,
        parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.topic = topic
        self._get_intrinsics = get_intrinsics
        self._get_paired_color = get_paired_color
        self._get_robot_state = get_robot_state
        self._get_camera_frame_id = get_camera_frame_id
        self._resolve_segment_camera_frame = resolve_segment_camera_frame
        self._get_tf_buffer = get_tf_buffer
        self._on_segment_pose = on_segment_pose
        self._on_click_uv = on_click_uv
        self._status_callback = status_callback
        self._frame_count = 0
        self._last_fps_time = _eai_platform_bootstrap.time.time()
        self._last_display_time = 0.0
        self._last_robot_overlay_time = 0.0
        self._fps = 0.0
        self._pending_seed_preview_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]] = None
        self._pending_seed_full_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]] = None
        self._point_count = 0
        self._latest_depth: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._intrinsics: _eai_platform_bootstrap.Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._depth_preview_raw: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._depth_full_shape: _eai_platform_bootstrap.Tuple[int, int] = (0, 0)
        self._preview_shape: _eai_platform_bootstrap.Tuple[int, int] = (0, 0)
        self._preview_intrinsics: _eai_platform_bootstrap.Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._pose_bridge = _eai_adapters_ros_bridges.PoseComputeBridge()
        self._pose_bridge.finished.connect(self._on_pose_compute_finished)
        self._depth_viz_bridge = _eai_adapters_ros_bridges.DepthVizBridge()
        self._depth_viz_bridge.finished.connect(self._on_depth_viz_finished)
        self._pose_busy = False
        self._depth_viz_busy = False
        self._depth_viz_pending = False
        self._gl_ready = False
        self.view = None
        self.scatter = None
        self.segment_scatter = None
        self.pose_obb_lines = None
        self.pose_axes_lines = None
        self.robot_left_scatter = None
        self.robot_right_scatter = None
        self.robot_left_axes = None
        self.robot_right_axes = None

        self.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
        layout = _eai_platform_bootstrap.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        title = topic.split("/")[-1] or topic
        self._title_base = title
        self.title_label = _eai_platform_bootstrap.QLabel(f"{title}  [depth]")
        self.title_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_TITLE, _eai_platform_bootstrap.QFont.Bold))
        self.title_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.title_label.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Fixed)
        layout.addWidget(self.title_label)

        gl_bar = _eai_platform_bootstrap.QHBoxLayout()
        gl_bar.setContentsMargins(0, 0, 0, 0)
        self._gl_enable_btn = _eai_platform_bootstrap.QPushButton("启用 3D 点云")
        self._gl_enable_btn.setToolTip(
            "默认仅显示 2D 深度图。多路 depth 同时开 OpenGL 易崩溃，故按需启用且全局仅一路。"
        )
        self._gl_enable_btn.clicked.connect(self._on_enable_gl_clicked)
        self._gl_disable_btn = _eai_platform_bootstrap.QPushButton("关闭 3D")
        self._gl_disable_btn.setVisible(False)
        self._gl_disable_btn.clicked.connect(self._teardown_gl)
        gl_bar.addWidget(self._gl_enable_btn)
        gl_bar.addWidget(self._gl_disable_btn)
        gl_bar.addStretch(1)
        layout.addLayout(gl_bar)

        self._gl_container = _eai_platform_bootstrap.QWidget()
        self._gl_container.setVisible(False)
        self._gl_container.setMinimumSize(80, 60)
        self._gl_container.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
        self._gl_container_layout = _eai_platform_bootstrap.QVBoxLayout(self._gl_container)
        self._gl_container_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gl_container, stretch=3)

        self.depth_preview = _eai_ui_widgets_images.ClickableImageLabel("点击深度图选择提示点；再按「调用分割」确认")
        self.depth_preview.setMinimumHeight(48)
        self.depth_preview.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
        self.depth_preview.clicked_pixel.connect(self._on_depth_pixel_clicked)
        layout.addWidget(self.depth_preview, stretch=2)

        self.info_label = _eai_platform_bootstrap.QLabel("等待深度图...")
        self.info_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        self.info_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.info_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        self.info_label.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Fixed)
        layout.addWidget(self.info_label)

        self.coord_label = _eai_platform_bootstrap.QLabel("点击深度图选择提示点；再点「调用分割 / SAM3 / FP」执行")
        self.coord_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        self.coord_label.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.coord_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE_BRIGHT};")
        self.coord_label.setWordWrap(True)
        self.coord_label.setMaximumHeight(40)
        self.coord_label.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Fixed)
        layout.addWidget(self.coord_label)

        robot_group = _eai_platform_bootstrap.QGroupBox("机器人：左右手 / 手臂 / 关节")
        robot_group.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Preferred, _eai_platform_bootstrap.QSizePolicy.Fixed)
        robot_group.setMaximumHeight(148)
        robot_group.setStyleSheet(
            "QGroupBox {"
            "  color: #e8e8e8;"
            "  font-weight: bold;"
            f"  border: 1px solid {_eai_configuration_policies.UI_BORDER};"
            "  border-radius: 4px;"
            "  margin-top: 6px;"
            "  padding-top: 6px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  left: 8px;"
            "  padding: 0 4px;"
            "  color: #ffffff;"
            "}"
        )
        robot_layout = _eai_platform_bootstrap.QVBoxLayout(robot_group)
        robot_layout.setContentsMargins(6, 4, 6, 4)
        self.robot_info_label = _eai_platform_bootstrap.QLabel("等待机器人 joint_states / TCP 数据...")
        self.robot_info_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_NORMAL))
        self.robot_info_label.setWordWrap(True)
        self.robot_info_label.setTextFormat(_eai_platform_bootstrap.Qt.RichText)
        self.robot_info_label.setStyleSheet(
            f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY};"
            f"background-color: {_eai_configuration_policies.UI_BG_LOG};"
            "padding: 4px 6px;"
            "border-radius: 3px;"
        )
        self.robot_info_label.setAlignment(_eai_platform_bootstrap.Qt.AlignTop | _eai_platform_bootstrap.Qt.AlignLeft)
        robot_layout.addWidget(self.robot_info_label)
        layout.addWidget(robot_group)

        self._robot_timer = _eai_platform_bootstrap.QTimer(self)
        self._robot_timer.timeout.connect(self._update_robot_overlay)
        self._robot_timer.start(int(_eai_configuration_policies.UI_ROBOT_STATE_MIN_INTERVAL_S * 1000))

    def stop_robot_timer(self) -> None:
        self._robot_timer.stop()

    def shutdown(self) -> None:
        """销毁前释放 OpenGL，避免无 context 绘制导致崩溃。"""
        try:
            self.stop_robot_timer()
        except Exception:
            pass
        self._teardown_gl()

    def _on_enable_gl_clicked(self) -> None:
        other = DepthPanel3D._active_gl_panel
        if other is not None and other is not self:
            try:
                other._teardown_gl()
            except Exception:
                pass
        if self._ensure_gl_view():
            self._request_depth_viz()

    def _ensure_gl_view(self) -> bool:
        if self._gl_ready and self.view is not None:
            return True
        try:
            _pg, _gl = _eai_platform_bootstrap._ensure_pyqtgraph_opengl()
            view = _gl.GLViewWidget()
            view.setMinimumSize(80, 60)
            view.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
            view.setBackgroundColor((30, 30, 30))
            view.opts["distance"] = 2.0

            grid = _gl.GLGridItem()
            grid.setSize(2, 2)
            grid.setSpacing(0.2, 0.2)
            view.addItem(grid)

            scatter = _gl.GLScatterPlotItem(
                pos=_eai_platform_bootstrap.np.zeros((1, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=_eai_platform_bootstrap.np.array([[0.5, 0.5, 0.5, 1.0]], dtype=_eai_platform_bootstrap.np.float32),
                size=3,
                pxMode=True,
            )
            view.addItem(scatter)

            segment_scatter = _gl.GLScatterPlotItem(
                pos=_eai_platform_bootstrap.np.zeros((1, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=_eai_platform_bootstrap.np.array([[1.0, 0.85, 0.1, 1.0]], dtype=_eai_platform_bootstrap.np.float32),
                size=6,
                pxMode=True,
            )
            view.addItem(segment_scatter)

            pose_obb_lines = _gl.GLLinePlotItem(
                pos=_eai_platform_bootstrap.np.zeros((2, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=(0.2, 0.9, 1.0, 1.0),
                width=2,
                antialias=True,
                mode="lines",
            )
            view.addItem(pose_obb_lines)

            pose_axes_lines = _gl.GLLinePlotItem(
                pos=_eai_platform_bootstrap.np.zeros((2, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=(1.0, 1.0, 1.0, 1.0),
                width=3,
                antialias=True,
                mode="lines",
            )
            view.addItem(pose_axes_lines)

            robot_left_scatter = _gl.GLScatterPlotItem(
                pos=_eai_platform_bootstrap.np.zeros((1, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=_eai_platform_bootstrap.np.array([[1.0, 0.2, 0.8, 1.0]], dtype=_eai_platform_bootstrap.np.float32),
                size=14,
                pxMode=False,
            )
            robot_right_scatter = _gl.GLScatterPlotItem(
                pos=_eai_platform_bootstrap.np.zeros((1, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=_eai_platform_bootstrap.np.array([[0.2, 0.85, 1.0, 1.0]], dtype=_eai_platform_bootstrap.np.float32),
                size=14,
                pxMode=False,
            )
            robot_left_axes = _gl.GLLinePlotItem(
                pos=_eai_platform_bootstrap.np.zeros((2, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=(1.0, 0.3, 0.8, 1.0),
                width=2,
                antialias=True,
                mode="lines",
            )
            robot_right_axes = _gl.GLLinePlotItem(
                pos=_eai_platform_bootstrap.np.zeros((2, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=(0.3, 0.85, 1.0, 1.0),
                width=2,
                antialias=True,
                mode="lines",
            )
            view.addItem(robot_left_scatter)
            view.addItem(robot_right_scatter)
            view.addItem(robot_left_axes)
            view.addItem(robot_right_axes)

            while self._gl_container_layout.count():
                item = self._gl_container_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

            self.view = view
            self.scatter = scatter
            self.segment_scatter = segment_scatter
            self.pose_obb_lines = pose_obb_lines
            self.pose_axes_lines = pose_axes_lines
            self.robot_left_scatter = robot_left_scatter
            self.robot_right_scatter = robot_right_scatter
            self.robot_left_axes = robot_left_axes
            self.robot_right_axes = robot_right_axes
            self._gl_container_layout.addWidget(view)
            self._gl_container.setVisible(True)
            self._gl_ready = True
            DepthPanel3D._active_gl_panel = self
            self._gl_enable_btn.setVisible(False)
            self._gl_disable_btn.setVisible(True)
            self.title_label.setText(f"{self._title_base}  [3D]")
            return True
        except Exception as exc:
            self._gl_ready = False
            if self._status_callback:
                self._status_callback(f"[{self.topic}] 启用 3D 失败: {exc}")
            return False

    def _teardown_gl(self) -> None:
        if DepthPanel3D._active_gl_panel is self:
            DepthPanel3D._active_gl_panel = None
        self._gl_ready = False
        view = self.view
        self.view = None
        self.scatter = None
        self.segment_scatter = None
        self.pose_obb_lines = None
        self.pose_axes_lines = None
        self.robot_left_scatter = None
        self.robot_right_scatter = None
        self.robot_left_axes = None
        self.robot_right_axes = None
        try:
            self._gl_container.setVisible(False)
        except Exception:
            pass
        try:
            self._gl_enable_btn.setVisible(True)
            self._gl_disable_btn.setVisible(False)
            self.title_label.setText(f"{self._title_base}  [depth]")
        except Exception:
            pass
        if view is not None:
            try:
                view.hide()
            except Exception:
                pass
            try:
                view.setParent(None)
            except Exception:
                pass
            try:
                view.deleteLater()
            except Exception:
                pass

    def _safe_gl_set(self, fn: _eai_platform_bootstrap.Callable[[], None]) -> None:
        if not self._gl_ready or self.view is None:
            return
        try:
            if not self.view.isVisible():
                return
            fn()
        except Exception:
            pass

    def _clear_robot_gl_items(self) -> None:
        if not self._gl_ready:
            return

        def _do() -> None:
            hidden = _eai_platform_bootstrap.np.zeros((1, 3), dtype=_eai_platform_bootstrap.np.float32)
            transparent = _eai_platform_bootstrap.np.array([[1.0, 1.0, 1.0, 0.0]], dtype=_eai_platform_bootstrap.np.float32)
            self.robot_left_scatter.setData(pos=hidden, color=transparent, size=14, pxMode=False)
            self.robot_right_scatter.setData(pos=hidden, color=transparent, size=14, pxMode=False)
            self.robot_left_axes.setData(pos=hidden, color=(1.0, 0.3, 0.8, 0.0), width=2, mode="lines")
            self.robot_right_axes.setData(pos=hidden, color=(0.3, 0.85, 1.0, 0.0), width=2, mode="lines")

        self._safe_gl_set(_do)

    def _set_robot_tcp_gl(
        self,
        scatter_item: _eai_platform_bootstrap.Any,
        axes_item: _eai_platform_bootstrap.Any,
        cam_pose: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.np.ndarray, _eai_platform_bootstrap.np.ndarray]],
        color_rgba: _eai_platform_bootstrap.Tuple[float, float, float, float],
    ) -> None:
        if not self._gl_ready or scatter_item is None or axes_item is None:
            return
        if cam_pose is None:
            def _hide() -> None:
                hidden = _eai_platform_bootstrap.np.zeros((1, 3), dtype=_eai_platform_bootstrap.np.float32)
                scatter_item.setData(
                    pos=hidden,
                    color=_eai_platform_bootstrap.np.array([[color_rgba[0], color_rgba[1], color_rgba[2], 0.0]], dtype=_eai_platform_bootstrap.np.float32),
                    size=14,
                    pxMode=False,
                )
                axes_item.setData(pos=hidden, color=(*color_rgba[:3], 0.0), width=2, mode="lines")

            self._safe_gl_set(_hide)
            return
        center, rotation = cam_pose
        if center[2] <= 0.02:
            self._set_robot_tcp_gl(scatter_item, axes_item, None, color_rgba)
            return

        def _show() -> None:
            scatter_item.setData(
                pos=center.reshape(1, 3),
                color=_eai_platform_bootstrap.np.array([color_rgba], dtype=_eai_platform_bootstrap.np.float32),
                size=14,
                pxMode=False,
            )
            axis_lines = _eai_adapters_robot_geometry.make_tcp_axis_lines(center, rotation, scale=0.07)
            axes_item.setData(pos=axis_lines, color=color_rgba, width=2, mode="lines")

        self._safe_gl_set(_show)

    def _update_robot_overlay(self) -> None:
        if self._get_robot_state is None or self._pose_busy:
            return
        now = _eai_platform_bootstrap.time.time()
        if now - self._last_robot_overlay_time < _eai_configuration_policies.UI_ROBOT_STATE_MIN_INTERVAL_S:
            return
        self._last_robot_overlay_time = now
        state = self._get_robot_state()
        camera_frame = ""
        if self._get_camera_frame_id is not None:
            camera_frame = self._get_camera_frame_id(self.topic)
        tf_buffer = self._get_tf_buffer() if self._get_tf_buffer else None

        left_cam = _eai_adapters_robot_geometry.transform_tcp_to_camera(state.left_tcp, camera_frame, tf_buffer)
        right_cam = _eai_adapters_robot_geometry.transform_tcp_to_camera(state.right_tcp, camera_frame, tf_buffer)

        left_xyz = left_cam[0] if left_cam is not None else None
        right_xyz = right_cam[0] if right_cam is not None else None
        self.robot_info_label.setText(
            _eai_adapters_robot_geometry.format_robot_state_html(state, camera_frame, left_xyz, right_xyz)
        )

        if not self._gl_ready:
            return
        self._set_robot_tcp_gl(
            self.robot_left_scatter,
            self.robot_left_axes,
            left_cam,
            (1.0, 0.25, 0.75, 1.0),
        )
        self._set_robot_tcp_gl(
            self.robot_right_scatter,
            self.robot_right_axes,
            right_cam,
            (0.25, 0.85, 1.0, 1.0),
        )

    def invoke_segment_at_seed(
        self,
        u_full: _eai_platform_bootstrap.Optional[int] = None,
        v_full: _eai_platform_bootstrap.Optional[int] = None,
    ) -> str:
        """按提示点启动立体分割+位姿（按钮触发）。"""
        if self._latest_depth is None:
            return f"[{self.topic}] 尚无深度图"
        if self._pose_busy:
            return f"[{self.topic}] 分割进行中，请稍候"
        if u_full is None or v_full is None:
            if self._pending_seed_full_uv is None:
                return f"[{self.topic}] 请先点击深度图选择提示点"
            u_full, v_full = self._pending_seed_full_uv
        dh, dw = self._depth_full_shape
        u_full = max(0, min(dw - 1, int(u_full)))
        v_full = max(0, min(dh - 1, int(v_full)))
        preview_u, preview_v = _eai_perception_image_geometry.scale_uv_to_shape(
            u_full, v_full, (dh, dw), self._preview_shape
        )
        self._pending_seed_full_uv = (u_full, v_full)
        self._pending_seed_preview_uv = (preview_u, preview_v)
        self.depth_preview.set_segment_overlay(
            None, seed_uv=(float(preview_u), float(preview_v))
        )
        self._start_pose_compute(u_full, v_full, preview_u, preview_v)
        return f"[{self.topic}] 已开始分割 @ depth({u_full}, {v_full})"

    def _start_pose_compute(self, u_full: int, v_full: int, preview_u: int, preview_v: int) -> None:
        if self._latest_depth is None or self._pose_busy:
            return
        self._pose_busy = True
        self.coord_label.setText("立体分割计算中...")
        fx, fy, cx, cy = self._intrinsics
        preview_intrinsics = self._preview_intrinsics
        get_color = self._get_paired_color
        depth_ref = self._latest_depth

        def _work() -> None:
            overlay = None
            try:
                depth_copy = depth_ref.copy()
                color_copy = None
                if get_color is not None:
                    color = get_color(self.topic)
                    if color is not None:
                        color_copy = color.copy()
                result, info = _eai_perception_pose_estimation.run_stereo_pose_pipeline_resampled(
                    depth_copy, u_full, v_full, fx, fy, cx, cy, color_bgr=color_copy
                )
                if result is not None:
                    overlay_source = color_copy
                    overlay_intr = preview_intrinsics
                    if overlay_source is None:
                        overlay_source = _eai_perception_image_geometry.build_depth_preview_image(depth_copy)
                        overlay_intr = preview_intrinsics
                    overlay = _eai_perception_image_geometry.build_pose_overlay_image(
                        overlay_source,
                        result,
                        overlay_intr,
                        max_dim=_eai_configuration_perception.SEGMENT_OVERLAY_MAX_DIM,
                    )
            except Exception as exc:
                result, info, overlay = None, f"点击 ({preview_u}, {preview_v})  |  计算错误: {exc}", None
            self._pose_bridge.finished.emit(result, info, preview_u, preview_v, overlay)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_depth_pixel_clicked(self, u: int, v: int) -> None:
        if self._latest_depth is None:
            return
        dh, dw = self._depth_full_shape
        u_full, v_full = _eai_perception_image_geometry.scale_uv_to_shape(u, v, self._preview_shape, (dh, dw))
        self._pending_seed_preview_uv = (u, v)
        self._pending_seed_full_uv = (u_full, v_full)
        self.depth_preview.set_segment_overlay(None, seed_uv=(float(u), float(v)))
        self.coord_label.setText(
            f"提示点 preview({u}, {v}) / depth({u_full}, {v_full})  |  "
            "请点「调用分割 / SAM3 / FP」确认"
        )
        if self._on_click_uv is not None:
            # 将深度点击映射到配对彩色图坐标，供 SAM3/FP 提示点使用
            color_u, color_v = u_full, v_full
            color_topic = self.topic
            if self._get_paired_color is not None:
                color = self._get_paired_color(self.topic)
                if color is not None:
                    color_u, color_v = _eai_perception_image_geometry.scale_uv_to_shape(
                        u_full, v_full, (dh, dw), color.shape[:2]
                    )
            self._on_click_uv(color_topic, int(color_u), int(color_v))
        if self._status_callback:
            self._status_callback(
                f"[{self.topic}] 已选提示点 depth({u_full}, {v_full})"
            )

    def _on_pose_compute_finished(
        self,
        result: _eai_platform_bootstrap.Optional[_eai_perception_pose_estimation.Object6DPoseResult],
        info: str,
        preview_u: int,
        preview_v: int,
        overlay: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
    ) -> None:
        self._pose_busy = False
        if self._depth_viz_pending:
            self._depth_viz_pending = False
            self._request_depth_viz()
        if result is None:
            self.depth_preview.clear_segment_overlay()
            self._clear_pose_visualization()
        else:
            mask_p = _eai_perception_image_geometry.resize_mask_to_shape(result.mask, self._preview_shape)
            cu, cv_pt = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(result.centroid_uv[0])),
                int(round(result.centroid_uv[1])),
                self._depth_full_shape,
                self._preview_shape,
            )
            contact_u, contact_v = _eai_perception_image_geometry.scale_uv_to_shape(
                int(round(result.contact_uv[0])),
                int(round(result.contact_uv[1])),
                self._depth_full_shape,
                self._preview_shape,
            )
            self.depth_preview.set_segment_overlay(
                mask_p,
                (float(cu), float(cv_pt)),
                obb_corners=result.obb_corners,
                intrinsics=self._preview_intrinsics,
                contact_uv=(float(contact_u), float(contact_v)),
                seed_uv=(float(preview_u), float(preview_v)),
                pose_position=_eai_platform_bootstrap.np.asarray(result.position_xyz, dtype=_eai_platform_bootstrap.np.float32),
                pose_rotation=_eai_platform_bootstrap.np.asarray(result.rotation_matrix, dtype=_eai_platform_bootstrap.np.float32),
                pose_axis_len=float(max(result.obb_extents) * 0.55),
            )
            self._show_pose_6d(result)
            self._emit_segment_pose(result)

        self.coord_label.setText(info)
        if self._status_callback:
            self._status_callback(f"[{self.topic}] {info.splitlines()[0]}")

    def _emit_segment_pose(self, result: _eai_perception_pose_estimation.Object6DPoseResult) -> None:
        if self._on_segment_pose is None:
            return
        if self._resolve_segment_camera_frame is not None:
            camera_frame = self._resolve_segment_camera_frame(self.topic, self.topic)
        elif self._get_camera_frame_id is not None:
            camera_frame = self._get_camera_frame_id(self.topic)
        else:
            return
        if not camera_frame:
            hint = (
                f"[{self.topic}] 分割成功但缺少相机 frame_id，"
                "无法用于左臂移动（请确认 depth 已发布 header.frame_id）"
            )
            if self._status_callback:
                self._status_callback(hint)
            return
        target = _eai_perception_pose_estimation.SegmentPoseTarget.from_pose_result(result, camera_frame, self.topic)
        _eai_platform_bootstrap.QTimer.singleShot(0, lambda t=target: self._on_segment_pose(t))

    def _clear_pose_visualization(self) -> None:
        if not self._gl_ready:
            return

        def _do() -> None:
            self.segment_scatter.setData(
                pos=_eai_platform_bootstrap.np.zeros((1, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=_eai_platform_bootstrap.np.array([[1.0, 0.85, 0.1, 0.0]], dtype=_eai_platform_bootstrap.np.float32),
                size=6,
                pxMode=True,
            )
            self.pose_obb_lines.setData(
                pos=_eai_platform_bootstrap.np.zeros((2, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=(0.2, 0.9, 1.0, 0.0),
                width=2,
                mode="lines",
            )
            self.pose_axes_lines.setData(
                pos=_eai_platform_bootstrap.np.zeros((2, 3), dtype=_eai_platform_bootstrap.np.float32),
                color=(1.0, 1.0, 1.0, 0.0),
                width=3,
                mode="lines",
            )

        self._safe_gl_set(_do)

    def _show_pose_6d(self, result: _eai_perception_pose_estimation.Object6DPoseResult) -> None:
        if not self._gl_ready:
            return

        def _do() -> None:
            points = _eai_perception_image_geometry.subsample_points(result.points_3d, _eai_configuration_policies.MAX_GL_SEGMENT_POINTS)
            if len(points) > 0:
                colors = _eai_platform_bootstrap.np.tile(
                    _eai_platform_bootstrap.np.array([1.0, 0.85, 0.1, 1.0], dtype=_eai_platform_bootstrap.np.float32), (len(points), 1)
                )
                self.segment_scatter.setData(pos=points, color=colors, size=5, pxMode=True)

            obb_lines = _eai_perception_pose_estimation.obb_wireframe_edges(result.obb_corners)
            self.pose_obb_lines.setData(
                pos=obb_lines,
                color=(0.2, 0.9, 1.0, 1.0),
                width=2,
                mode="lines",
            )

            axis_scale = max(float(max(result.obb_extents)) * 0.6, 0.04)
            center = _eai_platform_bootstrap.np.array(result.obb_center, dtype=_eai_platform_bootstrap.np.float32)
            axis_lines, _axis_colors = _eai_perception_pose_estimation.pose_axes_lines(
                center, result.rotation_matrix, axis_scale
            )
            self.pose_axes_lines.setData(
                pos=axis_lines,
                color=(1.0, 0.75, 0.2, 1.0),
                width=3,
                mode="lines",
            )

        self._safe_gl_set(_do)

    def _request_depth_viz(self) -> None:
        if self._latest_depth is None or self._pose_busy:
            return
        now = _eai_platform_bootstrap.time.time()
        if now - self._last_display_time < _eai_configuration_policies.UI_DEPTH_MIN_INTERVAL_S:
            return
        if self._depth_viz_busy:
            self._depth_viz_pending = True
            return
        self._depth_viz_busy = True
        self._last_display_time = now
        snap = self._latest_depth
        fx, fy, cx, cy = self._intrinsics
        need_cloud = bool(self._gl_ready)

        def _work() -> None:
            try:
                if need_cloud:
                    result = _eai_perception_image_geometry.build_depth_viz_data(snap.copy(), fx, fy, cx, cy)
                else:
                    # 2D 预览：只做伪彩，不做点云（多路 depth 时点云会卡死 UI）
                    preview_vis = _eai_perception_image_geometry.build_depth_preview_image(snap)
                    h, w = snap.shape[:2]
                    ph, pw = preview_vis.shape[:2]
                    result = _eai_perception_image_geometry.DepthVizResult(
                        preview_vis=preview_vis,
                        preview_shape=(ph, pw),
                        preview_intrinsics=(
                            fx * (pw / w),
                            fy * (ph / h),
                            cx * (pw / w),
                            cy * (ph / h),
                        ),
                        points=_eai_platform_bootstrap.np.zeros((0, 3), dtype=_eai_platform_bootstrap.np.float32),
                        colors=_eai_platform_bootstrap.np.zeros((0, 4), dtype=_eai_platform_bootstrap.np.float32),
                        point_count=0,
                        full_shape=(h, w),
                    )
            except Exception:
                result = None
            self._depth_viz_bridge.finished.emit(result)

        _eai_platform_bootstrap.threading.Thread(target=_work, daemon=True).start()

    def _on_depth_viz_finished(self, result: _eai_platform_bootstrap.Optional[_eai_perception_image_geometry.DepthVizResult]) -> None:
        self._depth_viz_busy = False
        pending = self._depth_viz_pending
        self._depth_viz_pending = False
        if result is not None and not self._pose_busy:
            self._preview_shape = result.preview_shape
            self._preview_intrinsics = result.preview_intrinsics
            self._depth_preview_raw = result.preview_vis
            self.depth_preview.set_source_image(result.preview_vis)
            self._point_count = result.point_count
            if self._gl_ready and self._point_count > 0:

                def _do() -> None:
                    self.scatter.setData(
                        pos=result.points, color=result.colors, size=3, pxMode=True
                    )
                    center = result.points.mean(axis=0)
                    _pg, _ = _eai_platform_bootstrap._ensure_pyqtgraph_opengl()
                    self.view.opts["center"] = _pg.Vector(center[0], center[1], center[2])

                self._safe_gl_set(_do)
            h, w = result.full_shape
            mode = "3D" if self._gl_ready else "2D"
            self.info_label.setText(
                f"{self.topic}  |  {w}x{h}  |  {self._point_count} pts  |  "
                f"{self._fps:.1f} Hz  |  {mode}"
            )
        if pending and not self._pose_busy:
            self._request_depth_viz()

    def update_depth(self, depth: _eai_platform_bootstrap.np.ndarray) -> None:
        h, w = depth.shape[:2]
        if self._get_intrinsics is not None:
            fx, fy, cx, cy = self._get_intrinsics(self.topic, w, h)
        else:
            fx, fy, cx, cy = _eai_perception_image_geometry.default_intrinsics(w, h)
        self._intrinsics = (fx, fy, cx, cy)
        self._depth_full_shape = (h, w)
        self._latest_depth = depth

        self._frame_count += 1
        now = _eai_platform_bootstrap.time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

        if not self._pose_busy:
            self._request_depth_viz()
        else:
            self._depth_viz_pending = True


def create_camera_panel(
    topic: str,
    get_intrinsics: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str, int, int], _eai_platform_bootstrap.Tuple[float, float, float, float]]] = None,
    get_paired_depth: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]]] = None,
    get_paired_color: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray]]] = None,
    get_depth_topic: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], _eai_platform_bootstrap.Optional[str]]] = None,
    is_paired_depth_enabled: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], bool]] = None,
    get_robot_state: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[], _eai_adapters_robot_geometry.RobotStateSnapshot]] = None,
    get_camera_frame_id: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], str]] = None,
    resolve_segment_camera_frame: _eai_platform_bootstrap.Optional[
        _eai_platform_bootstrap.Callable[[str, _eai_platform_bootstrap.Optional[str]], str]
    ] = None,
    get_tf_buffer: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[], _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Buffer]]] = None,
    on_segment_pose: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[_eai_perception_pose_estimation.SegmentPoseTarget], None]] = None,
    on_click_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str, int, int], None]] = None,
    status_callback: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Callable[[str], None]] = None,
) -> _eai_platform_bootstrap.QWidget:
    if _eai_perception_image_geometry.is_depth_topic(topic):
        return DepthPanel3D(
            topic,
            get_intrinsics=get_intrinsics,
            get_paired_color=get_paired_color,
            get_robot_state=get_robot_state,
            get_camera_frame_id=get_camera_frame_id,
            resolve_segment_camera_frame=resolve_segment_camera_frame,
            get_tf_buffer=get_tf_buffer,
            on_segment_pose=on_segment_pose,
            on_click_uv=on_click_uv,
            status_callback=status_callback,
        )
    return CameraPanel(
        topic,
        get_paired_depth=get_paired_depth,
        get_intrinsics_for_depth=get_intrinsics,
        get_depth_topic=get_depth_topic,
        get_camera_frame_id=get_camera_frame_id,
        resolve_segment_camera_frame=resolve_segment_camera_frame,
        is_paired_depth_enabled=is_paired_depth_enabled,
        on_segment_pose=on_segment_pose,
        on_click_uv=on_click_uv,
        status_callback=status_callback,
    )

