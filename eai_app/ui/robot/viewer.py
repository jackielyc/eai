from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.robot.dependencies')
from eai_app.ui.robot.interaction import InteractionMixin
from eai_app.ui.robot.sensor_layers import SensorLayersMixin
from eai_app.ui.robot.presentation import PresentationMixin

class RobotView3D(InteractionMixin, SensorLayersMixin, PresentationMixin, _deps.QWidget):
    """可嵌进手臂页的机器人视图。勾选要显示的深度。set_running(False) 时停止刷新。"""
    def __init__(self, parent=None, ros_node=None, preferences_path=None) -> None:
        super().__init__(parent)
        self._ros_node = ros_node
        layout = _deps.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = _deps.QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(_deps.QLabel("深度"))
        self._depth_on = {"head": True, "left": True, "right": True}
        self._depth_checks = {}
        for _topic, key, label, tip in _deps._DEPTH_KEYS:
            box = _deps.QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(tip)
            box.toggled.connect(lambda checked, name=key: self._set_depth_enabled(name, checked))
            row.addWidget(box)
            self._depth_checks[key] = box
        self._skel_check = _deps.QCheckBox("手骨架")
        self._skel_check.setChecked(True)
        self._skel_check.setToolTip("用当前手腕位姿和睿研手的开合量，在左右手上画出估计骨架。")
        self._skel_check.toggled.connect(self._set_skeleton_enabled)
        row.addWidget(self._skel_check)
        self._wrist_accessories_check = _deps.QCheckBox('腕部支架 / 相机')
        self._wrist_accessories_check.setChecked(True)
        self._wrist_accessories_check.setToolTip('显示随模型提供的腕部支架与 D405 外壳；按 URDF 安装关系随手腕移动。')
        self._wrist_accessories_check.toggled.connect(self._show_wrist_accessories)
        row.addWidget(self._wrist_accessories_check)
        self._pose_checks = {}
        for key, label, _rgb in _deps._POSE_METHODS:
            box = _deps.QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(f"画出最近一次 {label} 的包围盒和坐标轴。")
            box.toggled.connect(lambda checked, name=key: self._set_pose_enabled(name, checked))
            row.addWidget(box)
            self._pose_checks[key] = box
        self._sam3_check = _deps.QCheckBox("SAM3")
        self._sam3_check.setChecked(True)
        self._sam3_check.setToolTip("SAM3 掩码结合深度和相机标定生成的三维点云；二维原始掩码另存于结果目录。")
        self._sam3_check.setToolTip("画出最近一次估计里 SAM3 分出的物体点云。")
        self._sam3_check.toggled.connect(self._set_sam3_enabled)
        row.addWidget(self._sam3_check)
        self._lingbot_check = _deps.QCheckBox("LingBot")
        self._lingbot_check.setChecked(True)
        self._lingbot_check.setToolTip(
            "画出最近一次 LingBot-Depth 点云，挂在对应相机上。可取消勾选。"
        )
        self._lingbot_check.toggled.connect(self._set_lingbot_enabled)
        row.addWidget(self._lingbot_check)
        self._wrench_check = _deps.QCheckBox("六维力")
        self._wrench_check.setChecked(True)
        self._wrench_check.setToolTip(
            "在左右手腕 Link7 上画出 /hal/*_arm_data 的力（橙）和力矩（青），以及力的 RGB 分量。"
        )
        self._wrench_check.toggled.connect(self._set_wrench_enabled)
        row.addWidget(self._wrench_check)
        self._target_poses_check = _deps.QCheckBox("多目标6D")
        self._target_poses_check.setChecked(True)
        self._target_poses_check.setToolTip("同时显示多目标估计的包围盒和坐标轴，可取消勾选。")
        self._target_poses_check.toggled.connect(self._set_target_poses_enabled)
        row.addWidget(self._target_poses_check)
        self._collision_scene_check=_deps.QCheckBox('碰撞场景 / 附着物')
        self._collision_scene_check.setChecked(True)
        self._collision_scene_check.toggled.connect(lambda _checked:self.set_collision_scene(*self._collision_scene_hold))
        row.addWidget(self._collision_scene_check)
        row.addStretch(1)
        layout.addLayout(row)
        self._wrench_readout = _deps.QLabel("六维力：等待 /hal/left_arm_data、/hal/right_arm_data")
        self._wrench_readout.setTextFormat(_deps.Qt.RichText)
        self._wrench_readout.setWordWrap(True)
        self._wrench_readout.setStyleSheet(
            "color: #e8eaed; background: #1e1f22; padding: 4px 8px;"
            "border-radius: 4px; font-family: monospace;"
        )
        layout.addWidget(self._wrench_readout)
        self._status = _deps.QLabel("正在加载机器人模型…")
        self._status.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(self._status)
        reach_row = _deps.QHBoxLayout()
        self._reach_checks = {}
        for side, label in (('l', '左臂可达采样（青）'), ('r', '右臂可达采样（橙）')):
            box = _deps.QCheckBox(label)
            box.setToolTip('当前腰部姿态下，按 URDF 关节限位采样的 Link7 位置；不含避障及指定手朝向约束。')
            box.toggled.connect(self._refresh_reach_visibility)
            reach_row.addWidget(box)
            self._reach_checks[side] = box
        layout.addLayout(reach_row)
        self._reach_note = _deps.QLabel('勾选显示模型估算可达点；不是抓取配置工作空间，也不代表无碰撞可达。')
        self._reach_note.setWordWrap(True)
        layout.addWidget(self._reach_note)
        self._reach_items = {}
        self._reach_waist = None
        self._reach_job = None
        self._target_pose_item = None
        self._target_pose_hold = []
        self._simulation_q = None
        self._simulation_hands = {}
        self._simulation_label = _deps.QLabel('机器人姿态来源：真机反馈')
        layout.addWidget(self._simulation_label)
        self.grasp_action_callback = None
        actions = _deps.QHBoxLayout()
        self.grasp_stop_button = _deps.QPushButton('停止 / 取消')
        self.grasp_restore_button = _deps.QPushButton('恢复启动状态')
        self.grasp_stop_button.clicked.connect(lambda: self._grasp_action('stop'))
        self.grasp_restore_button.clicked.connect(lambda: self._grasp_action('restore'))
        actions.addWidget(self.grasp_stop_button)
        actions.addWidget(self.grasp_restore_button)
        self._restore_scene_check = _deps.QCheckBox('已确认恢复路径及手周围无障碍（真机）')
        actions.addWidget(self._restore_scene_check)
        self._grasp_action_note = _deps.QLabel('恢复双臂和双手；执行方式沿用分阶段抓取页，腰部不移动。')
        self._grasp_action_note.setWordWrap(True)
        layout.addLayout(actions)
        layout.addWidget(self._grasp_action_note)
        self._hand_model = None
        self.simulation_hand_callback = None
        self._sim_hand_controls = {}
        controls = _deps.QGridLayout()
        labels = ('拇指旋转', '拇指弯曲', '食指', '中指', '无名指', '小指')
        controls.addWidget(_deps.QLabel('仿真手控制（0 张开，1 闭合）'), 0, 0)
        for col, label in enumerate(labels, 1):
            controls.addWidget(_deps.QLabel(label), 0, col)
        for row_index, (side, label) in enumerate((('left', '左手'), ('right', '右手')), 1):
            controls.addWidget(_deps.QLabel(label), row_index, 0)
            boxes = []
            for col in range(6):
                box = _deps.QDoubleSpinBox()
                box.setRange(0., .9 if col < 2 else 1.)
                box.setSingleStep(.05)
                box.setDecimals(3)
                box.setEnabled(False)
                box.valueChanged.connect(lambda _value, side=side: self._change_simulation_hand(side))
                boxes.append(box)
                controls.addWidget(box, row_index, col+1)
            self._sim_hand_controls[side] = boxes
        layout.addLayout(controls)
        self._gl: _deps.Optional[_deps._RobotGL] = None
        self._kin: _deps.Optional[_deps._Kinematics] = None
        self._clouds: _deps.List[dict] = []
        self._cloud_by_topic: _deps.Dict[str, dict] = {}
        self._skels: _deps.List[dict] = []
        self._skel_on = True
        self._wrenches: _deps.List[dict] = []
        self._wrench_by_side: _deps.Dict[str, dict] = {}
        self._wrench_on = True
        self._pose_items: _deps.Dict[str, dict] = {}
        self._collision_scene_hold=({},[])
        self._collision_scene_items={}
        self._pose_on = {key: True for key, _label, _rgb in _deps._POSE_METHODS}
        self._pose_hold: _deps.Dict[str, tuple] = {}
        self._pose_note: _deps.Dict[str, str] = {}
        self._sam3: _deps.Optional[dict] = None
        self._sam3_on = True
        self._sam3_hold = None
        self._sam3_note = ""
        self._lingbot: _deps.Optional[dict] = None
        self._lingbot_on = True
        self._lingbot_hold = None
        self._lingbot_colors = None
        self._lingbot_parent = ""
        self._lingbot_T = _deps.np.eye(4, dtype=_deps.np.float64)
        self._lingbot_note = ""
        self._lock = _deps.threading.Lock()
        self._error = ""
        self._timer = _deps.QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._init_display_preferences(preferences_path)
        _deps.QTimer.singleShot(0, self._load)

def __getattr__(name):
    return getattr(_deps, name)

from eai_app.infrastructure.compat import forward_exports as _forward_exports
from eai_app.ui.robot import dependencies as _dependencies
_forward_exports(__name__, {name: _dependencies.__name__ for name in dir(_dependencies) if not name.startswith("__") and name not in globals()})
