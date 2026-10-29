"""ui.builders.context learning — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_video = module_ref('eai_app.ui.widgets.video')

def build(self, build):
    build.ctx_tab = _eai_platform_bootstrap.QWidget()
    ctx_outer = _eai_platform_bootstrap.QVBoxLayout(build.ctx_tab)
    ctx_outer.setContentsMargins(8, 6, 8, 6)
    ctx_outer.setSpacing(6)
    ctx_hint = _eai_platform_bootstrap.QLabel(
        "播放演示视频作为视觉参考；机器人「照着做」需配对的 .rrd 关节轨迹同步回放"
        "（当前栈无法仅从 RGB 视频推断动作）。选择视频后会自动尝试匹配同名 .rrd。"
    )
    ctx_hint.setWordWrap(True)
    ctx_hint.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    ctx_outer.addWidget(ctx_hint)

    self._ctx_video_path = ""
    self._ctx_rrd_path = ""

    ctx_ctrl = _eai_platform_bootstrap.QHBoxLayout()
    ctx_ctrl.setSpacing(6)
    self.ctx_select_video_btn = _eai_platform_bootstrap.QPushButton("选择视频")
    self.ctx_select_video_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ctx_select_video_btn.setToolTip(
        "选择演示 mp4/avi/mkv/webm。\n若同目录存在同名 .rrd，将自动配对。"
    )
    self.ctx_select_video_btn.clicked.connect(self._on_ctx_select_video_clicked)
    ctx_ctrl.addWidget(self.ctx_select_video_btn)
    self.ctx_select_rrd_btn = _eai_platform_bootstrap.QPushButton("选择 RRD")
    self.ctx_select_rrd_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ctx_select_rrd_btn.setToolTip(
        "选择与视频配对的 .rrd 轨迹；机器人跟做依赖此文件。"
    )
    self.ctx_select_rrd_btn.clicked.connect(self._on_ctx_select_rrd_clicked)
    ctx_ctrl.addWidget(self.ctx_select_rrd_btn)
    self.ctx_rerun_btn = _eai_platform_bootstrap.QPushButton("Rerun 播放")
    self.ctx_rerun_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ctx_rerun_btn.setEnabled(False)
    self.ctx_rerun_btn.setToolTip(
        "用 rerun-sdk 原生 Viewer 打开当前 .rrd（命令: rerun <file.rrd>）。\n"
        "仅可视化，不驱臂。可用 RERUN_BIN 指定可执行文件。"
    )
    self.ctx_rerun_btn.clicked.connect(self._on_ctx_rerun_clicked)
    ctx_ctrl.addWidget(self.ctx_rerun_btn)
    ctx_ctrl.addWidget(_eai_platform_bootstrap.QLabel("次数"))
    self.ctx_loop_spin = _eai_platform_bootstrap.QSpinBox()
    self.ctx_loop_spin.setRange(1, _eai_configuration_policies.REPLAY_LOOP_COUNT_MAX)
    self.ctx_loop_spin.setValue(1)
    self.ctx_loop_spin.setFixedWidth(52)
    self.ctx_loop_spin.setToolTip("RRD 回放次数；视频会循环播放以匹配")
    ctx_ctrl.addWidget(self.ctx_loop_spin)
    self.ctx_status_label = _eai_platform_bootstrap.QLabel("状态: 未开始")
    self.ctx_status_label.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ctx_ctrl.addWidget(self.ctx_status_label, 1)
    self.ctx_stack_btn = _eai_platform_bootstrap.QPushButton("启动机器人栈")
    self.ctx_stack_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ctx_stack_btn.setToolTip("与「回放」页相同：拉起 robot stack")
    self.ctx_stack_btn.clicked.connect(self._on_robot_stack_clicked)
    ctx_ctrl.addWidget(self.ctx_stack_btn)
    self.ctx_sync_start_btn = _eai_platform_bootstrap.QPushButton("同步开始")
    self.ctx_sync_start_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ctx_sync_start_btn.setToolTip(
        "同时播放视频并启动 RRD 回放，使机器人按轨迹跟做。\n"
        "先决条件：已选视频+.rrd + 手/臂服务栈运行中 + control_mode=0 + 手/臂已使能。"
    )
    self.ctx_sync_start_btn.clicked.connect(self._on_ctx_sync_start_clicked)
    ctx_ctrl.addWidget(self.ctx_sync_start_btn)
    self.ctx_stop_btn = _eai_platform_bootstrap.QPushButton("停止")
    self.ctx_stop_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    self.ctx_stop_btn.setObjectName("dangerAction")
    self.ctx_stop_btn.setEnabled(False)
    self.ctx_stop_btn.clicked.connect(self._on_ctx_stop_clicked)
    ctx_ctrl.addWidget(self.ctx_stop_btn)
    ctx_outer.addLayout(ctx_ctrl)

    ctx_paths = _eai_platform_bootstrap.QHBoxLayout()
    ctx_paths.setSpacing(6)
    self.ctx_video_path_edit = _eai_platform_bootstrap.QLineEdit()
    self.ctx_video_path_edit.setReadOnly(True)
    self.ctx_video_path_edit.setPlaceholderText("视频路径…")
    self.ctx_video_path_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ctx_paths.addWidget(self.ctx_video_path_edit, 1)
    self.ctx_rrd_path_edit = _eai_platform_bootstrap.QLineEdit()
    self.ctx_rrd_path_edit.setReadOnly(True)
    self.ctx_rrd_path_edit.setPlaceholderText("配对 RRD 路径（机器人跟做必需）…")
    self.ctx_rrd_path_edit.setFont(_eai_platform_bootstrap.QFont(_eai_configuration_policies.UI_MONO_FAMILY, _eai_configuration_policies.UI_MONO_SIZE_SMALL))
    ctx_paths.addWidget(self.ctx_rrd_path_edit, 1)
    ctx_outer.addLayout(ctx_paths)

    ctx_media_split = _eai_platform_bootstrap.QSplitter(_eai_platform_bootstrap.Qt.Horizontal)
    self.ctx_video_player = _eai_ui_widgets_video.VideoPlayerWidget(build.ctx_tab)
    self.ctx_video_player.setMinimumHeight(280)
    ctx_media_split.addWidget(self.ctx_video_player)

    self.ctx_rerun_host = _eai_platform_bootstrap.QWidget()
    ctx_rerun_layout = _eai_platform_bootstrap.QVBoxLayout(self.ctx_rerun_host)
    ctx_rerun_layout.setContentsMargins(0, 0, 0, 0)
    ctx_rerun_layout.setSpacing(4)
    self.ctx_rerun_title = _eai_platform_bootstrap.QLabel("Rerun（rerun-sdk）")
    self.ctx_rerun_title.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
    ctx_rerun_layout.addWidget(self.ctx_rerun_title)
    self.ctx_rerun_placeholder = _eai_platform_bootstrap.QLabel(
        "选择 .rrd 后点「Rerun 播放」，\n"
        "将用 rerun-sdk 原生 Viewer 打开该文件。\n\n"
        "等价命令: rerun <file.rrd>"
    )
    self.ctx_rerun_placeholder.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
    self.ctx_rerun_placeholder.setWordWrap(True)
    self.ctx_rerun_placeholder.setStyleSheet(
        "background-color: #111; color: #ccc; padding: 12px;"
    )
    self.ctx_rerun_placeholder.setMinimumHeight(280)
    ctx_rerun_layout.addWidget(self.ctx_rerun_placeholder, 1)
    ctx_media_split.addWidget(self.ctx_rerun_host)
    ctx_media_split.setStretchFactor(0, 1)
    ctx_media_split.setStretchFactor(1, 1)
    ctx_outer.addWidget(ctx_media_split, 1)

    self._load_ctx_default_video()
