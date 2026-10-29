"""ui.widgets.images — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_adapters_models_qwen_local = module_ref('eai_app.adapters.models.qwen_local')
_eai_adapters_robot_geometry = module_ref('eai_app.adapters.robot.geometry')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def resolve_test_scenario_image_path(title: str) -> _eai_platform_bootstrap.Optional[str]:
    """按场景名解析 images/ 下预览图路径；不存在则返回 None。"""
    candidates: _eai_platform_bootstrap.List[str] = []
    override = _eai_adapters_models_qwen_local.TEST_SCENARIO_IMAGE_FILES.get(title)
    if override:
        candidates.append(override)
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidates.append(f"{title}{ext}")
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        path = _eai_platform_bootstrap.os.path.join(_eai_configuration_policies.TEST_IMAGES_DIR, name)
        if _eai_platform_bootstrap.os.path.isfile(path):
            return path
    return None


class ScaledPixmapLabel(_eai_platform_bootstrap.QLabel):
    """保持宽高比缩放显示静态 QPixmap；可选点击选中 / 点选原图像素。"""

    clicked = _eai_platform_bootstrap.pyqtSignal()
    clicked_pixel = _eai_platform_bootstrap.pyqtSignal(int, int)

    def __init__(self, placeholder: str = "暂无图像", parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None,
                 *, fill_frame: bool = False) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.setMinimumSize(140, 100)
        self._fill_frame = bool(fill_frame)
        policy = _eai_platform_bootstrap.QSizePolicy.Ignored if self._fill_frame else _eai_platform_bootstrap.QSizePolicy.Expanding
        self.setSizePolicy(policy, policy)
        self._source_pixmap: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None
        self._source_wh: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]] = None
        self._selected = False
        self._clickable = False
        self._pick_pixel = False
        self._apply_frame_style()

    def set_clickable(self, enabled: bool) -> None:
        self._clickable = enabled
        if enabled or self._pick_pixel:
            self.setCursor(_eai_platform_bootstrap.Qt.PointingHandCursor)
        else:
            self.setCursor(_eai_platform_bootstrap.Qt.ArrowCursor)

    def set_pick_pixel(self, enabled: bool) -> None:
        """开启后左键点击映射到源图像素并 emit clicked_pixel(u, v)。"""
        self._pick_pixel = bool(enabled)
        if enabled or self._clickable:
            self.setCursor(_eai_platform_bootstrap.Qt.CrossCursor if enabled else _eai_platform_bootstrap.Qt.PointingHandCursor)
        else:
            self.setCursor(_eai_platform_bootstrap.Qt.ArrowCursor)

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self._apply_frame_style()

    def _apply_frame_style(self) -> None:
        border = "2px solid #4CAF50" if self._selected else "1px solid #555"
        self.setStyleSheet(
            f"QLabel {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; border: {border}; "
            f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; }}"
        )

    def set_source_pixmap(
        self,
        pixmap: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap],
        *,
        source_wh: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]] = None,
    ) -> None:
        self._source_pixmap = pixmap
        if source_wh is not None:
            self._source_wh = (int(source_wh[0]), int(source_wh[1]))
        elif pixmap is not None and not pixmap.isNull():
            self._source_wh = (int(pixmap.width()), int(pixmap.height()))
        else:
            self._source_wh = None
        if pixmap is None or pixmap.isNull():
            self.clear()
            self.setText("暂无图像")
        else:
            self.setText("")
            self._refresh_scaled()

    def _map_click_to_source(self, pos: _eai_platform_bootstrap.QPoint) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]]:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return None
        if self._source_wh is not None:
            src_w, src_h = self._source_wh
        else:
            src_w = int(self._source_pixmap.width())
            src_h = int(self._source_pixmap.height())
        if src_w <= 0 or src_h <= 0:
            return None
        displayed = self.pixmap()
        if displayed is None or displayed.isNull():
            return None
        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = displayed.width(), displayed.height()
        offset_x = (label_w - pix_w) // 2
        offset_y = (label_h - pix_h) // 2
        lx = pos.x() - offset_x
        ly = pos.y() - offset_y
        if lx < 0 or ly < 0 or lx >= pix_w or ly >= pix_h:
            return None
        u = int(lx * src_w / float(max(1, pix_w)))
        v = int(ly * src_h / float(max(1, pix_h)))
        u = max(0, min(src_w - 1, u))
        v = max(0, min(src_h - 1, v))
        return u, v

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == _eai_platform_bootstrap.Qt.LeftButton:
            if self._pick_pixel:
                uv = self._map_click_to_source(event.pos())
                if uv is not None:
                    self.clicked_pixel.emit(uv[0], uv[1])
                    return
            if self._clickable:
                self.clicked.emit()
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._fill_frame:
            # The source pixmap is already preview-sized and FastTransformation
            # is cheap. Immediate scaling avoids visible staircase growth.
            self._refresh_scaled()
            return
        # Other large preview users retain debouncing to avoid tab-switch stalls.
        timer = getattr(self, "_resize_timer", None)
        if timer is None:
            timer = _eai_platform_bootstrap.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._refresh_scaled)
            self._resize_timer = timer
        timer.start(80)

    def sizeHint(self) -> _eai_platform_bootstrap.QSize:  # noqa: N802
        if self._fill_frame:
            return _eai_platform_bootstrap.QSize(320, 180)
        return super().sizeHint()

    def minimumSizeHint(self) -> _eai_platform_bootstrap.QSize:  # noqa: N802
        if self._fill_frame:
            return _eai_platform_bootstrap.QSize(140, 100)
        return super().minimumSizeHint()

    def _refresh_scaled(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            _eai_platform_bootstrap.QSize(max(1, self.width()), max(1, self.height())),
            _eai_platform_bootstrap.Qt.KeepAspectRatio,
            _eai_platform_bootstrap.Qt.FastTransformation,
        )
        super().setPixmap(scaled)


class Mv6dLightboxWindow(_eai_platform_bootstrap.QWidget):
    """多视角 6D 头/左腕/右腕渲染大图（顶层弹出，可随估计结果刷新）。"""

    closed_by_user = _eai_platform_bootstrap.pyqtSignal()

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(parent, _eai_platform_bootstrap.Qt.Window | _eai_platform_bootstrap.Qt.WindowStaysOnTopHint)
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("多视角 6D 预览")
        self.setMinimumSize(960, 360)
        self.resize(1400, 520)
        root = _eai_platform_bootstrap.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = _eai_platform_bootstrap.QHBoxLayout()
        top.setSpacing(8)
        self.error_label = _eai_platform_bootstrap.QLabel("误差: --")
        self.error_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
        self.error_label.setStyleSheet(f"color: {_eai_configuration_policies.UI_ACCENT_BLUE};")
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(
            _eai_platform_bootstrap.Qt.TextSelectableByMouse | _eai_platform_bootstrap.Qt.TextSelectableByKeyboard
        )
        self.error_label.setFocusPolicy(_eai_platform_bootstrap.Qt.ClickFocus)
        self.error_label.setCursor(_eai_platform_bootstrap.Qt.IBeamCursor)
        top.addWidget(self.error_label, 1)
        close_btn = _eai_platform_bootstrap.QPushButton("关闭")
        close_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        close_btn.setFixedWidth(72)
        close_btn.clicked.connect(self.close)
        top.addWidget(close_btn)
        root.addLayout(top)

        views = _eai_platform_bootstrap.QHBoxLayout()
        views.setSpacing(8)
        self.head_label = ScaledPixmapLabel("头部 6D", fill_frame=True)
        self.left_wrist_label = ScaledPixmapLabel("左腕 6D", fill_frame=True)
        self.right_wrist_label = ScaledPixmapLabel("右腕 6D", fill_frame=True)
        # 兼容旧调用：wrist_label 指向左腕
        self.wrist_label = self.left_wrist_label
        for lab, title in (
            (self.head_label, "头部"),
            (self.left_wrist_label, "左腕"),
            (self.right_wrist_label, "右腕"),
        ):
            lab.setMinimumSize(280, 220)
            lab.setStyleSheet(
                f"QLabel {{ background-color: #111; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
                f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; }}"
            )
            lab.setToolTip(f"{title} 6D 渲染（随估计刷新）")
            views.addWidget(lab, 1)
        root.addLayout(views, 1)
        self._user_closed = False

    def set_error_text(self, text: str) -> None:
        self.error_label.setText(text or "误差: --")

    def set_views(
        self,
        head_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap],
        left_wrist_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None,
        right_wrist_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None,
        *,
        wrist_pix: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None,
    ) -> None:
        """刷新三视角。兼容旧签名：仅传 head + wrist 时填到左腕。"""
        if self._user_closed:
            return
        if wrist_pix is not None and left_wrist_pix is None and right_wrist_pix is None:
            left_wrist_pix = wrist_pix
        for lab, pix, empty in (
            (self.head_label, head_pix, "头部 6D（无帧）"),
            (self.left_wrist_label, left_wrist_pix, "左腕 6D（无帧）"),
            (self.right_wrist_label, right_wrist_pix, "右腕 6D（无帧）"),
        ):
            if pix is not None and not pix.isNull():
                lab.set_source_pixmap(pix)
            else:
                lab.set_source_pixmap(None)
                lab.setText(empty)

    def closeEvent(self, event: _eai_platform_bootstrap.QCloseEvent) -> None:  # type: ignore[override]
        if not self._user_closed:
            self._user_closed = True
            self.closed_by_user.emit()
        super().closeEvent(event)


class ClickableImageLabel(_eai_platform_bootstrap.QLabel):
    """可点击图像，将控件坐标映射回原始像素。"""

    clicked_pixel = _eai_platform_bootstrap.pyqtSignal(int, int)

    def __init__(self, placeholder: str = "等待图像...", parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        self.setMinimumSize(64, 48)
        self.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
        self.setStyleSheet(
            f"background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: {_eai_configuration_policies.UI_TEXT_MUTED}; border: 1px solid {_eai_configuration_policies.UI_BORDER};"
        )
        self.setCursor(_eai_platform_bootstrap.Qt.CrossCursor)
        self._source_image: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._latest_pixmap: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None
        self._segment_mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._segment_centroid: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None
        self._segment_contact_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None
        self._segment_seed_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None
        self._segment_obb: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._segment_intrinsics: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]] = None
        self._pose_position: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._pose_rotation: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None
        self._pose_axis_len: _eai_platform_bootstrap.Optional[float] = None
        # 持久数字条（随视频流每帧重绘）
        self._pose_hud: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any]] = None

    def set_source_image(self, cv_image: _eai_platform_bootstrap.np.ndarray, pixmap: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None) -> None:
        self._source_image = cv_image
        self._refresh_display(pixmap)

    def set_segment_overlay(
        self,
        mask: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray],
        centroid_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
        obb_corners: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        intrinsics: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float, float, float]] = None,
        contact_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
        seed_uv: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[float, float]] = None,
        pose_position: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        pose_rotation: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.np.ndarray] = None,
        pose_axis_len: _eai_platform_bootstrap.Optional[float] = None,
        pose_hud: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any]] = None,
    ) -> None:
        self._segment_mask = mask.copy() if mask is not None else None
        self._segment_centroid = centroid_uv
        self._segment_contact_uv = contact_uv
        self._segment_seed_uv = seed_uv
        self._segment_obb = obb_corners.copy() if obb_corners is not None else None
        self._segment_intrinsics = intrinsics
        self._pose_position = (
            _eai_platform_bootstrap.np.asarray(pose_position, dtype=_eai_platform_bootstrap.np.float32).reshape(3).copy()
            if pose_position is not None
            else None
        )
        self._pose_rotation = (
            _eai_platform_bootstrap.np.asarray(pose_rotation, dtype=_eai_platform_bootstrap.np.float32).reshape(3, 3).copy()
            if pose_rotation is not None
            else None
        )
        self._pose_axis_len = pose_axis_len
        if pose_hud is not None:
            self._pose_hud = dict(pose_hud)
        elif pose_position is None:
            # 非 6D 叠加（如纯 SAM mask）时清掉旧数字条，避免残留
            self._pose_hud = None
        self._refresh_display()

    def set_pose_hud(self, hud: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Dict[str, _eai_platform_bootstrap.Any]]) -> None:
        self._pose_hud = dict(hud) if hud else None
        self._refresh_display()

    def clear_segment_overlay(self) -> None:
        self._segment_mask = None
        self._segment_centroid = None
        self._segment_contact_uv = None
        self._segment_seed_uv = None
        self._segment_obb = None
        self._segment_intrinsics = None
        self._pose_position = None
        self._pose_rotation = None
        self._pose_axis_len = None
        self._pose_hud = None
        self._refresh_display()

    def set_precomposed_display(self, overlay_bgr: _eai_platform_bootstrap.np.ndarray) -> None:
        """直接显示后台线程已合成好的叠加图，避免 UI 线程重复计算。"""
        # 保留为源图，这样后续帧更新前至少还能看到一次；
        # 真正持久化请用 set_segment_overlay（随视频流重绘）。
        self._source_image = overlay_bgr.copy()
        self._segment_mask = None
        self._segment_centroid = None
        self._segment_contact_uv = None
        self._segment_seed_uv = None
        self._segment_obb = None
        self._segment_intrinsics = None
        self._pose_position = None
        self._pose_rotation = None
        self._pose_axis_len = None
        self._pose_hud = None
        self._latest_pixmap = _eai_adapters_robot_geometry.cv2_to_qpixmap(overlay_bgr)
        self._render_pixmap()

    def _compose_display_image(self) -> _eai_platform_bootstrap.np.ndarray:
        if self._source_image is None:
            return _eai_platform_bootstrap.np.zeros((1, 1, 3), dtype=_eai_platform_bootstrap.np.uint8)
        has_mask = self._segment_mask is not None and _eai_platform_bootstrap.np.asarray(self._segment_mask).any()
        has_pose = self._segment_obb is not None or self._pose_position is not None
        has_seed = self._segment_seed_uv is not None
        if has_mask or has_pose or has_seed:
            display = _eai_adapters_robot_geometry.apply_segment_overlay(
                self._source_image,
                self._segment_mask,
                self._segment_centroid,
                obb_corners=self._segment_obb,
                intrinsics=self._segment_intrinsics,
                contact_uv=self._segment_contact_uv,
                seed_uv=self._segment_seed_uv,
                pose_position=self._pose_position,
                pose_rotation=self._pose_rotation,
                pose_axis_len=self._pose_axis_len,
            )
        else:
            display = self._source_image
        hud = self._pose_hud
        if hud and self._pose_position is not None and self._pose_rotation is not None:
            rpy = _eai_perception_pose_estimation.rotation_matrix_to_euler_xyz(
                _eai_platform_bootstrap.np.asarray(self._pose_rotation, dtype=_eai_platform_bootstrap.np.float64)
            )
            _eai_adapters_robot_geometry.draw_mv6d_hud_banner(
                display,
                title=str(hud.get("title") or "6D"),
                xyz=self._pose_position,
                rpy_rad=rpy,
                err_t_mm=hud.get("err_t_mm"),
                err_r_deg=hud.get("err_r_deg"),
                method=str(hud.get("method") or ""),
            )
        elif hud and hud.get("xyz") is not None:
            xyz = hud.get("xyz")
            rpy = hud.get("rpy_rad") or (0.0, 0.0, 0.0)
            _eai_adapters_robot_geometry.draw_mv6d_hud_banner(
                display,
                title=str(hud.get("title") or "6D"),
                xyz=xyz,
                rpy_rad=rpy,
                err_t_mm=hud.get("err_t_mm"),
                err_r_deg=hud.get("err_r_deg"),
                method=str(hud.get("method") or ""),
            )
        return display

    def _refresh_display(self, pixmap: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPixmap] = None) -> None:
        if self._source_image is None:
            return
        overlay_active = (
            self._segment_mask is not None
            or self._segment_seed_uv is not None
            or self._segment_obb is not None
            or self._pose_position is not None
            or self._pose_hud is not None
        )
        if pixmap is not None and not overlay_active:
            self._latest_pixmap = pixmap
        else:
            self._latest_pixmap = _eai_adapters_robot_geometry.cv2_to_qpixmap(self._compose_display_image())
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._latest_pixmap is None:
            return
        target = self.size()
        if target.width() < 16 or target.height() < 16:
            self.setPixmap(self._latest_pixmap)
            return
        # Fast：实时预览用；Smooth 会把主线程打满 → 「无响应」
        scaled = self._latest_pixmap.scaled(
            target,
            _eai_platform_bootstrap.Qt.KeepAspectRatio,
            _eai_platform_bootstrap.Qt.FastTransformation,
        )
        self.setPixmap(scaled)

    def _map_click_to_image(self, pos: _eai_platform_bootstrap.QPoint) -> _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.Tuple[int, int]]:
        if self._source_image is None or self._latest_pixmap is None:
            return None

        label_w, label_h = self.width(), self.height()
        pix_w, pix_h = self._latest_pixmap.width(), self._latest_pixmap.height()
        scale = min(label_w / pix_w, label_h / pix_h)
        disp_w = int(pix_w * scale)
        disp_h = int(pix_h * scale)
        offset_x = (label_w - disp_w) // 2
        offset_y = (label_h - disp_h) // 2

        lx, ly = pos.x() - offset_x, pos.y() - offset_y
        if lx < 0 or ly < 0 or lx >= disp_w or ly >= disp_h:
            return None

        img_h, img_w = self._source_image.shape[:2]
        u = int(lx / scale)
        v = int(ly / scale)
        u = max(0, min(img_w - 1, u))
        v = max(0, min(img_h - 1, v))
        return u, v

    def mousePressEvent(self, event: _eai_platform_bootstrap.QMouseEvent) -> None:
        if event.button() == _eai_platform_bootstrap.Qt.LeftButton:
            uv = self._map_click_to_image(event.pos())
            if uv is not None:
                self.clicked_pixel.emit(uv[0], uv[1])
        super().mousePressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        t = getattr(self, "_resize_timer", None)
        if t is None:
            t = _eai_platform_bootstrap.QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(self._render_pixmap)
            self._resize_timer = t
        t.start(80)

