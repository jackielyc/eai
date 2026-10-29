from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.grasp.dependencies')
from eai_app.ui.grasp.configuration_controller import ConfigurationControllerMixin
from eai_app.ui.grasp.execution_controller import ExecutionControllerMixin
from eai_app.ui.grasp.perception_controller import PerceptionControllerMixin
from eai_app.ui.grasp.workflow_controller import WorkflowControllerMixin
from eai_app.ui.grasp.presentation import PresentationMixin

class GraspExecutionPanel(ConfigurationControllerMixin, ExecutionControllerMixin, PerceptionControllerMixin, WorkflowControllerMixin, PresentationMixin, _deps.GraspRuntimeMixin, _deps.QWidget):
    def __init__(self, node, busy, parent=None, preview_image=None, open_perception=None,
                 fill_extrinsic=None, ensure_feedback=None, open_calibration=None, robot_preview=None,
                 simulation_display=None, wrench_feedback=None, ensure_wrench_monitor=None,
                 save_task_data=None, load_task_data=None, verify_observation=None, scene_display=None,
                 reestimate_observation=None, open_robot=None, assembly_live_frame=None):
        super().__init__(parent)
        self.node, self.other_busy = node, busy
        self.preview_image_callback = preview_image
        self.robot_preview_callback = robot_preview
        self.simulation_display = simulation_display
        self.wrench_feedback = wrench_feedback
        self.ensure_wrench_monitor = ensure_wrench_monitor
        self.save_task_data = save_task_data
        self.load_task_data = load_task_data
        self.scene_display_callback=scene_display
        self.reestimate_callback=reestimate_observation
        self.open_robot_callback=open_robot
        self.assembly_live_frame_callback=assembly_live_frame
        self._program_active=False;self._program_backend_authorized=False
        self._program_validator=None;self._program_action_result=None;self._program_observation_role=None
        self._last_execution_mode='simulation'
        self._wrench_guard_triggered = None
        self.target_observations = {}
        self._workflow_step = 0
        self._workflow_substep = 0
        self._workflow_finished = False
        self._workflow_completed_steps = set()
        self._workflow_waiting = False
        self.hand_runner = None
        self._workflow_confirmation = None
        self._workflow_error = ''
        self._workflow_step_settings = {}
        self._workflow_auto_running = False
        self._workflow_auto_paused = False
        self._workflow_auto_cancel_requested = False
        self._workflow_auto_restoring = False
        self._workflow_auto_initial_snapshot = None
        self._workflow_auto_timer = _deps.QTimer(self)
        self._workflow_auto_timer.setSingleShot(True)
        self._workflow_auto_timer.timeout.connect(self._run_auto_workflow_next)
        self.simulation = None
        self._sim_reserved = False
        self._execute_after_plan = False
        self._startup_snapshot = None
        self._startup_deadline = _deps.time.monotonic() + 10.
        self._startup_recording = True
        self._restoring = False
        self._restore_hands_goal = None
        try:
            self._startup_seed = _deps.read_feedback(_deps.os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))['q16']
        except Exception:
            self._startup_seed = [0.] * 16
        self.open_perception = open_perception
        self.fill_extrinsic = fill_extrinsic
        self.ensure_feedback = ensure_feedback
        self.open_calibration = open_calibration
        self.sequence = _deps.Sequence()
        self.observation = None
        self.method_observations = {}
        self.process = None
        self.runner = None
        self.mode = ''
        self.root = _deps.PROJECT_ROOT
        self.init_runtime(verify_observation)
        self.run_log = _deps.TaskRunLog(_deps.os.environ.get('EAI_GRASP_RUN_LOG_DIR', self.root / '.cache/grasp_runs'))
        self._run_log_error = ''
        self.job_directory = None
        self.message = '等待点击估计结果；这里只接受匹配实际手型的抓取候选'
        layout = _deps.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        heading = _deps.QLabel('分阶段抓取')
        heading.setStyleSheet('font-size: 17px; font-weight: 600;')
        layout.addWidget(heading)
        hint = _deps.QLabel('预览、手部验证和 IK 预检可独立使用；执行本段会自动预检所选动作，通过后执行。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        task_row = _deps.QHBoxLayout()
        task_row.addWidget(_deps.QLabel('任务'))
        self.task_combo = _deps.QComboBox()
        for key, task in _deps.TASKS.items():
            self.task_combo.addItem(task['name'], key)
        self.task_combo.currentIndexChanged.connect(self.change_task_workflow)
        task_row.addWidget(self.task_combo)
        self.task_new_button = _deps.QPushButton('新建任务…')
        self.task_new_button.clicked.connect(self.new_task)
        task_row.addWidget(self.task_new_button)
        self.task_copy_button = _deps.QPushButton('复制任务…')
        self.task_copy_button.clicked.connect(self.copy_task)
        task_row.addWidget(self.task_copy_button)
        self.task_manage_button = _deps.QPushButton('管理任务…')
        self.task_manage_button.clicked.connect(self.manage_tasks)
        task_row.addWidget(self.task_manage_button)
        self.workflow_next_button = _deps.QPushButton('执行流程下一步')
        self.workflow_next_button.clicked.connect(self.execute_workflow_next)
        task_row.addWidget(self.workflow_next_button)
        self.workflow_confirm_button = _deps.QPushButton('确认当前流程结果')
        self.workflow_confirm_button.clicked.connect(self.confirm_workflow_action)
        task_row.addWidget(self.workflow_confirm_button)
        self.workflow_reset_button = _deps.QPushButton('重置流程')
        self.workflow_reset_button.clicked.connect(self.reset_task_workflow)
        task_row.addWidget(self.workflow_reset_button)
        self.task_save_button = _deps.QPushButton('保存任务数据')
        self.task_save_button.clicked.connect(self.save_current_task)
        task_row.addWidget(self.task_save_button)
        self.workflow_all_button = _deps.QPushButton('全部执行')
        self.workflow_all_button.clicked.connect(self.toggle_all_workflow)
        task_row.addWidget(self.workflow_all_button)
        self.workflow_auto_pause_button = _deps.QPushButton('暂停')
        self.workflow_auto_pause_button.clicked.connect(self.toggle_auto_workflow_pause)
        self.workflow_auto_pause_button.setEnabled(False)
        task_row.addWidget(self.workflow_auto_pause_button)
        task_row.addStretch(1)
        layout.addLayout(task_row)
        self.task_flow = _deps.TaskFlowWidget()
        self.task_flow.step_selected.connect(self.select_workflow_step)
        layout.addWidget(self.task_flow)
        self.workflow_status_label = _deps.QLabel()
        self.workflow_status_label.setWordWrap(True)
        layout.addWidget(self.workflow_status_label)
        self.task_role_row = _deps.QHBoxLayout()
        self.task_role_combos = {}
        layout.addLayout(self.task_role_row)
        self.task_data_status = _deps.QLabel('任务数据：尚未保存')
        self.task_data_status.setWordWrap(True)
        layout.addWidget(self.task_data_status)
        mode_row = _deps.QHBoxLayout()
        mode_row.addWidget(_deps.QLabel('执行方式'))
        self.execution_mode = _deps.QComboBox()
        self.execution_mode.addItem('仿真执行（仅机器人页面）', 'simulation')
        self.execution_mode.addItem('真机执行', 'hardware')
        self.execution_mode.currentIndexChanged.connect(self.change_execution_mode)
        mode_row.addWidget(self.execution_mode)
        self.simulation_controls_button = _deps.QPushButton('打开仿真控制')
        self.simulation_controls_button.clicked.connect(self.open_simulation_controls)
        mode_row.addWidget(self.simulation_controls_button)
        layout.addLayout(mode_row)
        self.build_runtime_controls(layout)
        setup_row = _deps.QHBoxLayout()
        self.check_button = _deps.QPushButton('检查准备状态')
        self.check_button.clicked.connect(self.check_readiness)
        setup_row.addWidget(self.check_button)
        self.perception_button = _deps.QPushButton('前往视觉6D估计')
        self.perception_button.clicked.connect(self.go_perception)
        setup_row.addWidget(self.perception_button)
        self.feedback_button = _deps.QPushButton('启动 FK 反馈')
        self.feedback_button.clicked.connect(self.start_feedback)
        setup_row.addWidget(self.feedback_button)
        self.calibration_button = _deps.QPushButton('前往抓取标定')
        self.calibration_button.clicked.connect(self.go_calibration)
        setup_row.addWidget(self.calibration_button)
        layout.addLayout(setup_row)
        self.readiness_label = _deps.QLabel('先检查配置，再到「手动控制 → 多视角6D位姿」选中物体并估计；结果自动传入。')
        self.readiness_label.setWordWrap(True)
        self.readiness_label.setTextInteractionFlags(_deps.Qt.TextSelectableByMouse)
        layout.addWidget(self.readiness_label)
        row = _deps.QHBoxLayout()
        row.addWidget(_deps.QLabel('抓取配置'))
        self.profile_path = _deps.QLineEdit(_deps.os.environ.get('EAI_GRASP_PROFILE', str(self.root / 'config/grasp_execution.json')))
        row.addWidget(self.profile_path, 1)
        self.browse_profile_button = _deps.QPushButton('选择配置…')
        self.browse_profile_button.clicked.connect(self.browse_profile)
        row.addWidget(self.browse_profile_button)
        layout.addLayout(row)
        expiry_row = _deps.QHBoxLayout()
        expiry_row.addWidget(_deps.QLabel('规划有效期'))
        self.plan_validity = _deps.QDoubleSpinBox()
        self.plan_validity.setRange(.1, 3600)
        self.plan_validity.setDecimals(1)
        self.plan_validity.setSingleStep(60)
        self.plan_validity.setSuffix(' 秒')
        self.plan_validity.setValue(600)
        expiry_row.addWidget(self.plan_validity)
        self.save_validity_button = _deps.QPushButton('保存有效期')
        self.save_validity_button.clicked.connect(self.save_plan_validity)
        expiry_row.addWidget(self.save_validity_button)
        layout.addLayout(expiry_row)
        self.plan_expiry_label = _deps.QLabel('尚无已通过预检的计划；通过后开始倒计时')
        layout.addWidget(self.plan_expiry_label)
        target_row = _deps.QHBoxLayout()
        target_row.addWidget(_deps.QLabel('操作目标'))
        self.grasp_target_combo = _deps.QComboBox()
        self.grasp_target_combo.currentIndexChanged.connect(self._on_grasp_target_changed)
        target_row.addWidget(self.grasp_target_combo, 1)
        layout.addLayout(target_row)
        source_row = _deps.QHBoxLayout()
        source_row.addWidget(_deps.QLabel('位姿来源'))
        self.source_combo = _deps.QComboBox()
        for label, key in (('PCA', 'pca'), ('FoundationPose', 'foundationpose'), ('Any6D', 'any6d')):
            self.source_combo.addItem(label + '（本轮无结果）', key)
            self.source_combo.model().item(self.source_combo.count()-1).setEnabled(False)
        self.source_combo.currentIndexChanged.connect(self.select_observation_source)
        source_row.addWidget(self.source_combo)
        layout.addLayout(source_row)
        row = _deps.QHBoxLayout()
        row.addWidget(_deps.QLabel('抓取候选'))
        self.candidates = _deps.QComboBox()
        row.addWidget(self.candidates, 1)
        self.import_button = _deps.QPushButton('导入手型候选')
        self.import_button.clicked.connect(self.import_candidates)
        row.addWidget(self.import_button)
        layout.addLayout(row)
        manual_row = _deps.QHBoxLayout()
        layout.addWidget(_deps.QLabel('人工接触偏移（base_link，cm）'))
        self.contact_offsets = []
        for axis in 'XYZ':
            manual_row.addWidget(_deps.QLabel(axis))
            spin = _deps.QDoubleSpinBox()
            spin.setRange(-15, 15)
            spin.setDecimals(1)
            spin.setSingleStep(.5)
            self.contact_offsets.append(spin)
            manual_row.addWidget(spin)
        self.manual_candidate_button = _deps.QPushButton('按当前手朝向生成候选')
        self.manual_candidate_button.clicked.connect(self.make_manual_candidate)
        layout.addLayout(manual_row)
        layout.addWidget(self.manual_candidate_button)
        self.hand_shape_suggestion = None
        self.hand_shape_label = _deps.QLabel('物体手型建议：等待视觉6D结果')
        self.hand_shape_label.setWordWrap(True)
        layout.addWidget(self.hand_shape_label)
        shape_row = _deps.QHBoxLayout()
        self.suggest_shape_button = _deps.QPushButton('根据物体估计手型')
        self.suggest_shape_button.clicked.connect(self.show_shape_result)
        shape_row.addWidget(self.suggest_shape_button)
        self.apply_shape_button = _deps.QPushButton('采用已标定手型')
        self.apply_shape_button.clicked.connect(self.apply_shape)
        shape_row.addWidget(self.apply_shape_button)
        layout.addLayout(shape_row)
        self.scene_confirm = _deps.QCheckBox('已核对场景与现场一致，并确认路径及手周围没有未建模障碍')
        layout.addWidget(self.scene_confirm)
        stage_row = _deps.QHBoxLayout()
        stage_row.addWidget(_deps.QLabel('抓取手臂'))
        self.grasp_arm = _deps.QComboBox()
        self.grasp_arm.addItem('左臂 / Link7_l', 'left')
        self.grasp_arm.addItem('右臂 / Link7_r', 'right')
        self.grasp_arm.currentIndexChanged.connect(self.change_grasp_arm)
        stage_row.addWidget(self.grasp_arm)
        self.waist_participation = _deps.QCheckBox('腰部参与 IK')
        self.waist_participation.setChecked(False)
        self.waist_participation.setToolTip(
            '允许腰部升降和俯仰参与所选手臂的 IK 与执行；当前无自动碰撞规划。'
        )
        self.waist_participation.toggled.connect(self.change_waist_participation)
        stage_row.addWidget(self.waist_participation)
        stage_row.addWidget(_deps.QLabel('本段动作'))
        self.motion_stage = _deps.QComboBox()
        for label, key in (('预抓取', 'pregrasp'), ('接近物体', 'approach'), ('抬起', 'lift')):
            self.motion_stage.addItem(label, key)
        stage_row.addWidget(self.motion_stage)
        layout.addLayout(stage_row)
        wrench_row = _deps.QHBoxLayout()
        self.wrench_guard_check = _deps.QCheckBox('六维力超限自动暂停')
        self.wrench_guard_check.setChecked(True)
        self.wrench_guard_check.setToolTip(
            '真机运动时监控所选腕部合力/合力矩；数据缺失、过期或传感器异常也会暂停。'
        )
        wrench_row.addWidget(self.wrench_guard_check)
        wrench_row.addWidget(_deps.QLabel('合力上限'))
        self.force_limit = _deps.QDoubleSpinBox()
        self.force_limit.setRange(.1, 500.)
        self.force_limit.setDecimals(1)
        self.force_limit.setValue(15.)
        self.force_limit.setSuffix(' N')
        wrench_row.addWidget(self.force_limit)
        wrench_row.addWidget(_deps.QLabel('合力矩上限'))
        self.torque_limit = _deps.QDoubleSpinBox()
        self.torque_limit.setRange(.01, 50.)
        self.torque_limit.setDecimals(2)
        self.torque_limit.setValue(1.)
        self.torque_limit.setSuffix(' N·m')
        wrench_row.addWidget(self.torque_limit)
        layout.addLayout(wrench_row)
        self.wrench_guard_status = _deps.QLabel(
            '保护起始值未经3D打印支架破坏试验认证；请按材料、打印方向和实测载荷调整。'
        )
        self.wrench_guard_status.setWordWrap(True)
        layout.addWidget(self.wrench_guard_status)
        row = _deps.QGridLayout()
        self.preview_button = _deps.QPushButton('预览抓取位置')
        self.hand_button = _deps.QPushButton('张开手')
        self.confirm_button = _deps.QPushButton('确认手已张开')
        self.plan_button = _deps.QPushButton('预检本段 IK')
        self.execute_button = _deps.QPushButton('执行本段')
        self.pause_button = _deps.QPushButton('暂停')
        self.stop_button = _deps.QPushButton('停止 / 取消')
        self.restore_button = _deps.QPushButton('恢复启动状态')
        self.restore_button.setToolTip('恢复本次界面启动时的双臂关节和双手开合；腰部保持当前位置')
        for index, (button, callback) in enumerate(((self.preview_button, self.preview), (self.hand_button, self.hand),
                                 (self.confirm_button, self.confirm), (self.plan_button, self.plan),
                                 (self.execute_button, self.execute), (self.pause_button, self.pause_or_resume),
                                 (self.stop_button, self.stop), (self.restore_button, self.restore_startup))):
            button.clicked.connect(callback)
            button.setMinimumHeight(34)
            row.addWidget(button, index // 3, index % 3)
        self.stop_button.setStyleSheet('QPushButton { border: 1px solid #c65b55; font-weight: 600; }')
        layout.addLayout(row)
        self.execution_feedback = _deps.QLabel('尚未请求执行')
        self.execution_feedback.setWordWrap(True)
        self.execution_feedback.setStyleSheet('color: #d0a85c;')
        layout.addWidget(self.execution_feedback)
        self.report = _deps.QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(130)
        self.report.setMaximumHeight(260)
        layout.addWidget(self.report)
        self.preview_image = _deps.QLabel('抓取预览：等待选择候选')
        self.preview_image.setMaximumHeight(200)
        layout.addWidget(self.preview_image)
        self.watchdog = _deps.QTimer(self)
        self.watchdog.setInterval(20)
        self.watchdog.timeout.connect(self.monitor)
        self.watchdog.start()
        self.profile_path.textChanged.connect(self.invalidate_preview)
        self.profile_path.textChanged.connect(self.load_plan_validity)
        self.profile_path.textChanged.connect(lambda _text:self.check_readiness())
        self.load_plan_validity()
        self.candidates.currentIndexChanged.connect(self.invalidate_preview)
        for spin in self.contact_offsets:
            spin.valueChanged.connect(self.offset_changed)
        self.capture_startup_state()
        if self.ensure_wrench_monitor:
            self.ensure_wrench_monitor()
        self.change_task_workflow()
        self.check_readiness()
        self.refresh()
        if self.use_execution_service:
            from eai_app.tasks.phone_assembly.panel import install_phone_console
            install_phone_console(self,layout)

def __getattr__(name):
    return getattr(_deps, name)

from eai_app.infrastructure.compat import forward_exports as _forward_exports
from eai_app.ui.grasp import dependencies as _dependencies
_forward_exports(__name__, {name: _dependencies.__name__ for name in vars(_dependencies) if not name.startswith("__") and name not in globals()})
