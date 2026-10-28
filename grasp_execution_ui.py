"""Qt grasp control panel: explicit stages, no execution on pose arrival."""
from __future__ import annotations

import json
import copy
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget, QGridLayout,
                             QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QInputDialog)

from grasp_execution import (Sequence, JointRunner, evaluate_wrench, load_profile,
                             make_preview, read_feedback, manual_candidate)
from grasp_task_workflow import (TASKS, TaskFlowWidget, format_steps, new_task_key,
                                 parse_roles, parse_steps, reload_tasks, save_tasks,
                                 validate_task)


class TaskEditorDialog(QDialog):
    def __init__(self, source=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('新建抓取任务')
        self.resize(650, 500)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit('' if source is None else source['name']+' 副本')
        self.roles_edit = QLineEdit(
            'object=目标' if source is None else
            ', '.join(f'{key}={label}' for key, label in source['roles']))
        self.steps_edit = QTextEdit(
            '选择目标|target|object\n确认操作|confirm\n完成|complete'
            if source is None else format_steps(source))
        form.addRow('任务名称', self.name_edit)
        form.addRow('目标角色', self.roles_edit)
        form.addRow('流程步骤', self.steps_edit)
        layout.addLayout(form)
        hint = QLabel(
            '角色格式：role=显示名，多个用逗号分隔。\n'
            '步骤每行：名称|动作|角色|参数。动作支持 target、motion、hand、confirm、complete；'
            'motion 参数为 pregrasp/approach/lift，hand 参数为 open/close。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def definition(self):
        return validate_task({'name': self.name_edit.text().strip(), 'builtin': False,
                              'roles': parse_roles(self.roles_edit.text()),
                              'steps': parse_steps(self.steps_edit.toPlainText())})


class GraspExecutionPanel(QWidget):
    def __init__(self, node, busy, parent=None, preview_image=None, open_perception=None,
                 fill_extrinsic=None, ensure_feedback=None, open_calibration=None, robot_preview=None,
                 simulation_display=None, wrench_feedback=None, ensure_wrench_monitor=None,
                 save_task_data=None, load_task_data=None):
        super().__init__(parent)
        self.node, self.other_busy = node, busy
        self.preview_image_callback = preview_image
        self.robot_preview_callback = robot_preview
        self.simulation_display = simulation_display
        self.wrench_feedback = wrench_feedback
        self.ensure_wrench_monitor = ensure_wrench_monitor
        self.save_task_data = save_task_data
        self.load_task_data = load_task_data
        self._wrench_guard_triggered = None
        self.target_observations = {}
        self._workflow_step = 0
        self._workflow_substep = 0
        self._workflow_finished = False
        self._workflow_waiting = False
        self._workflow_step_settings = {}
        self._workflow_auto_running = False
        self._workflow_auto_paused = False
        self._workflow_auto_cancel_requested = False
        self._workflow_auto_restoring = False
        self._workflow_auto_initial_snapshot = None
        self._workflow_auto_timer = QTimer(self)
        self._workflow_auto_timer.setSingleShot(True)
        self._workflow_auto_timer.timeout.connect(self._run_auto_workflow_next)
        self.simulation = None
        self._sim_reserved = False
        self._execute_after_plan = False
        self._startup_snapshot = None
        self._startup_deadline = time.monotonic() + 10.
        self._startup_recording = True
        self._restoring = False
        self._restore_hands_goal = None
        try:
            self._startup_seed = read_feedback(os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))['q16']
        except Exception:
            self._startup_seed = [0.] * 16
        self.open_perception = open_perception
        self.fill_extrinsic = fill_extrinsic
        self.ensure_feedback = ensure_feedback
        self.open_calibration = open_calibration
        self.sequence = Sequence()
        self.observation = None
        self.method_observations = {}
        self.process = None
        self.runner = None
        self.mode = ''
        self.root = Path(__file__).resolve().parent
        self.job_directory = None
        self.message = '等待点击估计结果；这里只接受匹配实际手型的抓取候选'
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        heading = QLabel('分阶段抓取')
        heading.setStyleSheet('font-size: 17px; font-weight: 600;')
        layout.addWidget(heading)
        hint = QLabel('预览、手部验证和 IK 预检可独立使用；执行本段会自动预检所选动作，通过后执行。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel('任务'))
        self.task_combo = QComboBox()
        for key, task in TASKS.items():
            self.task_combo.addItem(task['name'], key)
        self.task_combo.currentIndexChanged.connect(self.change_task_workflow)
        task_row.addWidget(self.task_combo)
        self.task_new_button = QPushButton('新建任务…')
        self.task_new_button.clicked.connect(self.new_task)
        task_row.addWidget(self.task_new_button)
        self.task_copy_button = QPushButton('复制任务…')
        self.task_copy_button.clicked.connect(self.copy_task)
        task_row.addWidget(self.task_copy_button)
        self.task_manage_button = QPushButton('管理任务…')
        self.task_manage_button.clicked.connect(self.manage_tasks)
        task_row.addWidget(self.task_manage_button)
        self.workflow_next_button = QPushButton('执行流程下一步')
        self.workflow_next_button.clicked.connect(self.execute_workflow_next)
        task_row.addWidget(self.workflow_next_button)
        self.workflow_reset_button = QPushButton('重置流程')
        self.workflow_reset_button.clicked.connect(self.reset_task_workflow)
        task_row.addWidget(self.workflow_reset_button)
        self.task_save_button = QPushButton('保存任务数据')
        self.task_save_button.clicked.connect(self.save_current_task)
        task_row.addWidget(self.task_save_button)
        self.workflow_all_button = QPushButton('全部执行')
        self.workflow_all_button.clicked.connect(self.toggle_all_workflow)
        task_row.addWidget(self.workflow_all_button)
        self.workflow_auto_pause_button = QPushButton('暂停')
        self.workflow_auto_pause_button.clicked.connect(self.toggle_auto_workflow_pause)
        self.workflow_auto_pause_button.setEnabled(False)
        task_row.addWidget(self.workflow_auto_pause_button)
        task_row.addStretch(1)
        layout.addLayout(task_row)
        self.task_flow = TaskFlowWidget()
        self.task_flow.step_selected.connect(self.select_workflow_step)
        layout.addWidget(self.task_flow)
        self.task_role_row = QHBoxLayout()
        self.task_role_combos = {}
        layout.addLayout(self.task_role_row)
        self.task_data_status = QLabel('任务数据：尚未保存')
        self.task_data_status.setWordWrap(True)
        layout.addWidget(self.task_data_status)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('执行方式'))
        self.execution_mode = QComboBox()
        self.execution_mode.addItem('仿真执行（仅机器人页面）', 'simulation')
        self.execution_mode.addItem('真机执行', 'hardware')
        self.execution_mode.currentIndexChanged.connect(self.change_execution_mode)
        mode_row.addWidget(self.execution_mode)
        self.simulation_controls_button = QPushButton('打开仿真控制')
        self.simulation_controls_button.clicked.connect(self.open_simulation_controls)
        mode_row.addWidget(self.simulation_controls_button)
        layout.addLayout(mode_row)
        setup_row = QHBoxLayout()
        self.check_button = QPushButton('检查准备状态')
        self.check_button.clicked.connect(self.check_readiness)
        setup_row.addWidget(self.check_button)
        self.perception_button = QPushButton('前往视觉6D估计')
        self.perception_button.clicked.connect(self.go_perception)
        setup_row.addWidget(self.perception_button)
        self.feedback_button = QPushButton('启动 FK 反馈')
        self.feedback_button.clicked.connect(self.start_feedback)
        setup_row.addWidget(self.feedback_button)
        self.calibration_button = QPushButton('前往抓取标定')
        self.calibration_button.clicked.connect(self.go_calibration)
        setup_row.addWidget(self.calibration_button)
        layout.addLayout(setup_row)
        self.readiness_label = QLabel('先检查配置，再到「手动控制 → 多视角6D位姿」选中物体并估计；结果自动传入。')
        self.readiness_label.setWordWrap(True)
        self.readiness_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.readiness_label)
        row = QHBoxLayout()
        row.addWidget(QLabel('抓取配置'))
        self.profile_path = QLineEdit(os.environ.get('EAI_GRASP_PROFILE', str(self.root / 'config/grasp_execution.json')))
        row.addWidget(self.profile_path, 1)
        self.browse_profile_button = QPushButton('选择配置…')
        self.browse_profile_button.clicked.connect(self.browse_profile)
        row.addWidget(self.browse_profile_button)
        layout.addLayout(row)
        expiry_row = QHBoxLayout()
        expiry_row.addWidget(QLabel('规划有效期'))
        self.plan_validity = QDoubleSpinBox()
        self.plan_validity.setRange(.1, 3600)
        self.plan_validity.setDecimals(1)
        self.plan_validity.setSingleStep(60)
        self.plan_validity.setSuffix(' 秒')
        self.plan_validity.setValue(600)
        expiry_row.addWidget(self.plan_validity)
        self.save_validity_button = QPushButton('保存有效期')
        self.save_validity_button.clicked.connect(self.save_plan_validity)
        expiry_row.addWidget(self.save_validity_button)
        layout.addLayout(expiry_row)
        self.plan_expiry_label = QLabel('尚无已通过预检的计划；通过后开始倒计时')
        layout.addWidget(self.plan_expiry_label)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel('操作目标'))
        self.grasp_target_combo = QComboBox()
        self.grasp_target_combo.currentIndexChanged.connect(self._on_grasp_target_changed)
        target_row.addWidget(self.grasp_target_combo, 1)
        layout.addLayout(target_row)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel('位姿来源'))
        self.source_combo = QComboBox()
        for label, key in (('PCA', 'pca'), ('FoundationPose', 'foundationpose'), ('Any6D', 'any6d')):
            self.source_combo.addItem(label + '（本轮无结果）', key)
            self.source_combo.model().item(self.source_combo.count()-1).setEnabled(False)
        self.source_combo.currentIndexChanged.connect(self.select_observation_source)
        source_row.addWidget(self.source_combo)
        layout.addLayout(source_row)
        row = QHBoxLayout()
        row.addWidget(QLabel('抓取候选'))
        self.candidates = QComboBox()
        row.addWidget(self.candidates, 1)
        self.import_button = QPushButton('导入手型候选')
        self.import_button.clicked.connect(self.import_candidates)
        row.addWidget(self.import_button)
        layout.addLayout(row)
        manual_row = QHBoxLayout()
        layout.addWidget(QLabel('人工接触偏移（base_link，cm）'))
        self.contact_offsets = []
        for axis in 'XYZ':
            manual_row.addWidget(QLabel(axis))
            spin = QDoubleSpinBox()
            spin.setRange(-15, 15)
            spin.setDecimals(1)
            spin.setSingleStep(.5)
            self.contact_offsets.append(spin)
            manual_row.addWidget(spin)
        self.manual_candidate_button = QPushButton('按当前手朝向生成候选')
        self.manual_candidate_button.clicked.connect(self.make_manual_candidate)
        layout.addLayout(manual_row)
        layout.addWidget(self.manual_candidate_button)
        self.hand_shape_suggestion = None
        self.hand_shape_label = QLabel('物体手型建议：等待视觉6D结果')
        self.hand_shape_label.setWordWrap(True)
        layout.addWidget(self.hand_shape_label)
        shape_row = QHBoxLayout()
        self.suggest_shape_button = QPushButton('根据物体估计手型')
        self.suggest_shape_button.clicked.connect(self.show_shape_result)
        shape_row.addWidget(self.suggest_shape_button)
        self.apply_shape_button = QPushButton('采用已标定手型')
        self.apply_shape_button.clicked.connect(self.apply_shape)
        shape_row.addWidget(self.apply_shape_button)
        layout.addLayout(shape_row)
        self.scene_confirm = QCheckBox('手动分步模式：当前无自动避障，已确认本段路径及手周围无障碍')
        layout.addWidget(self.scene_confirm)
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel('抓取手臂'))
        self.grasp_arm = QComboBox()
        self.grasp_arm.addItem('左臂 / Link7_l', 'left')
        self.grasp_arm.addItem('右臂 / Link7_r', 'right')
        self.grasp_arm.currentIndexChanged.connect(self.change_grasp_arm)
        stage_row.addWidget(self.grasp_arm)
        self.waist_participation = QCheckBox('腰部参与 IK')
        self.waist_participation.setChecked(False)
        self.waist_participation.setToolTip(
            '允许腰部升降和俯仰参与所选手臂的 IK 与执行；当前无自动碰撞规划。'
        )
        self.waist_participation.toggled.connect(self.change_waist_participation)
        stage_row.addWidget(self.waist_participation)
        stage_row.addWidget(QLabel('本段动作'))
        self.motion_stage = QComboBox()
        for label, key in (('预抓取', 'pregrasp'), ('接近物体', 'approach'), ('抬起', 'lift')):
            self.motion_stage.addItem(label, key)
        stage_row.addWidget(self.motion_stage)
        layout.addLayout(stage_row)
        wrench_row = QHBoxLayout()
        self.wrench_guard_check = QCheckBox('六维力超限自动暂停')
        self.wrench_guard_check.setChecked(True)
        self.wrench_guard_check.setToolTip(
            '真机运动时监控所选腕部合力/合力矩；数据缺失、过期或传感器异常也会暂停。'
        )
        wrench_row.addWidget(self.wrench_guard_check)
        wrench_row.addWidget(QLabel('合力上限'))
        self.force_limit = QDoubleSpinBox()
        self.force_limit.setRange(.1, 500.)
        self.force_limit.setDecimals(1)
        self.force_limit.setValue(15.)
        self.force_limit.setSuffix(' N')
        wrench_row.addWidget(self.force_limit)
        wrench_row.addWidget(QLabel('合力矩上限'))
        self.torque_limit = QDoubleSpinBox()
        self.torque_limit.setRange(.01, 50.)
        self.torque_limit.setDecimals(2)
        self.torque_limit.setValue(1.)
        self.torque_limit.setSuffix(' N·m')
        wrench_row.addWidget(self.torque_limit)
        layout.addLayout(wrench_row)
        self.wrench_guard_status = QLabel(
            '保护起始值未经3D打印支架破坏试验认证；请按材料、打印方向和实测载荷调整。'
        )
        self.wrench_guard_status.setWordWrap(True)
        layout.addWidget(self.wrench_guard_status)
        row = QGridLayout()
        self.preview_button = QPushButton('预览抓取位置')
        self.hand_button = QPushButton('张开手')
        self.confirm_button = QPushButton('确认手已张开')
        self.plan_button = QPushButton('预检本段 IK')
        self.execute_button = QPushButton('执行本段')
        self.pause_button = QPushButton('暂停')
        self.stop_button = QPushButton('停止 / 取消')
        self.restore_button = QPushButton('恢复启动状态')
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
        self.execution_feedback = QLabel('尚未请求执行')
        self.execution_feedback.setWordWrap(True)
        self.execution_feedback.setStyleSheet('color: #d0a85c;')
        layout.addWidget(self.execution_feedback)
        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(130)
        self.report.setMaximumHeight(260)
        layout.addWidget(self.report)
        self.preview_image = QLabel('抓取预览：等待选择候选')
        self.preview_image.setMaximumHeight(200)
        layout.addWidget(self.preview_image)
        self.watchdog = QTimer(self)
        self.watchdog.setInterval(20)
        self.watchdog.timeout.connect(self.monitor)
        self.watchdog.start()
        self.profile_path.textChanged.connect(self.invalidate_preview)
        self.profile_path.textChanged.connect(self.load_plan_validity)
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

    def wrench_guard_violation(self, sides=None):
        if self.is_simulation or not self.wrench_guard_check.isChecked():
            self.wrench_guard_status.setText('仿真不产生真实六维力，保护不触发。' if self.is_simulation else '六维力保护已关闭。')
            return None
        if self.wrench_feedback is None:
            self.wrench_guard_status.setText('六维力保护不可用：未连接反馈读取器。')
            return '未连接六维力反馈读取器'
        if sides is None:
            sides = (str(self.grasp_arm.currentData() or 'right'),)
        reports = []
        for side in sides:
            sample = self.wrench_feedback(side)
            violation, force_norm, torque_norm = evaluate_wrench(
                sample, self.force_limit.value(), self.torque_limit.value())
            label = '左腕' if side == 'left' else '右腕'
            if force_norm is not None:
                reports.append(f'{label} |F|={force_norm:.2f}N |M|={torque_norm:.3f}N·m')
            if violation:
                self.wrench_guard_status.setText(label + '：' + violation)
                return label + '：' + violation
        self.wrench_guard_status.setText('；'.join(reports) if reports else '等待六维力反馈')
        return None

    def read_startup_hands(self):
        from arm_text_commands import HAND_JOINT_ORDER, canonical_hand_joint
        stamps = getattr(self.node, '_grasp_hand_state_received_at', {})
        if any(not 0 <= time.time()-stamps.get(side, 0) <= 2. for side in ('left','right')):
            raise ValueError('双手反馈尚未就绪或已过期')
        state = self.node.get_robot_state()
        hands = {}
        for side in ('left','right'):
            values = {canonical_hand_joint(name): float(value) for name,value in getattr(state, side+'_hand')}
            vector = [values[name] for name in HAND_JOINT_ORDER]
            from grasp_execution import hand_positions
            hands[side] = hand_positions(vector)
        return hands

    def capture_startup_state(self):
        if not self._startup_recording:
            return
        if self._startup_snapshot is None:
            try:
                measured = read_feedback(os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))
                self._startup_snapshot = copy.deepcopy(dict(measured, hands={}))
                self.simulation = None
            except Exception:
                measured = None
        if self._startup_snapshot is not None and not self._startup_snapshot.get('hands'):
            try:
                self._startup_snapshot['hands'] = self.read_startup_hands()
            except Exception:
                pass
        if self._startup_snapshot is not None and self._startup_snapshot.get('hands'):
            self._startup_recording = False
        elif time.monotonic() > self._startup_deadline:
            self._startup_recording = False

    def restore_startup(self):
        if self.process or self.runner or self.sequence.state == 'stop_unconfirmed':
            self.message = '请先停止当前任务并等待停止确认，再恢复启动状态。'
            self.refresh()
            return
        try:
            self._startup_recording = False
            if self.is_simulation:
                simulation = self.simulation_backend()
                startup = self._simulation_startup
            else:
                if self._startup_snapshot is None:
                    raise ValueError('界面启动时未采集到完整双臂/双手反馈，不能恢复；不会用零位代替')
                startup = self._startup_snapshot
            self._start_restore_snapshot(startup, auto_restore=False)
        except Exception as exc:
            self._restoring = False
            self.message = '恢复未启动：' + str(exc)
            self.execution_feedback.setText(self.message)
        self.refresh()

    def _start_restore_snapshot(self, startup, *, auto_restore=False):
        profile = self.selected_profile()
        measured = self.simulation_backend().feedback() if self.is_simulation else self.feedback()
        from grasp_restore import prepare_restore
        identifier = uuid.uuid4().hex
        plan = prepare_restore(profile, measured, startup, identifier)
        preview = dict(id=identifier, arm_side='both', hand_model=profile['hand_model'],
                       tcp_link='Link7_l / Link7_r', base_frame='base_link',
                       observation_id='restore:'+identifier, goals={'approach':plan['target']})
        self.sequence.reset(preview, profile)
        self.sequence.pending_stage = 'restore'
        self.sequence.plan = dict(plan, accepted_at=time.monotonic())
        self.sequence.state = 'planned'
        self._restore_hands_goal = copy.deepcopy(startup.get('hands') or {})
        self._restoring = True
        self._workflow_auto_restoring = bool(auto_restore)
        self._execute_ready_plan()
        if self.runner is None:
            self._restoring = False

    def finish_restore_hands(self):
        hands = self._restore_hands_goal
        if not hands:
            return '双臂已恢复启动状态；启动时未采集到双手反馈，双手恢复已跳过。'
        if self.is_simulation:
            for side, values in hands.items():
                self.simulation_backend().hand(side, values)
            return '仿真双臂和双手已恢复启动状态。'
        self.feedback()
        if not self.node.is_hand_enabled():
            raise ValueError('双臂已恢复，但手未使能，无法恢复双手')
        self.node._grasp_hand_command_authorized = True
        try:
            for side, values in hands.items():
                self.node.apply_hand_joint_positions(side, values)
        finally:
            self.node._grasp_hand_command_authorized = False
        return '双臂已恢复启动状态；双手恢复指令已发送，请观察实物确认到位。'

    def load_plan_validity(self, *_):
        if not self.reserved:
            self.simulation = None
        try:
            profile = load_profile(self.profile_path.text().strip(), require_calibration=False)
            self.plan_validity.setValue(float(profile['max_plan_age_s']))
            if hasattr(self, 'grasp_arm'):
                self.grasp_arm.blockSignals(True)
                self.grasp_arm.setCurrentIndex(0 if profile['arm_side'] == 'left' else 1)
                self.grasp_arm.blockSignals(False)
        except Exception:
            self.plan_validity.setValue(600)

    def change_grasp_arm(self, *_):
        if self.runner or self.process or self.reserved:
            return
        if (self.observation and
                self.observation.get('grasps', {}).get('model') == 'manual_current_hand_orientation'):
            self.observation.pop('grasps', None)
            self.candidates.clear()
        self.sequence = Sequence()
        self.scene_confirm.setChecked(False)
        side = '左臂' if self.grasp_arm.currentData() == 'left' else '右臂'
        self.message = f'已选择{side}抓取；依赖旧手腕朝向的候选和计划已清除，请重新生成或预览。'
        self.check_readiness()
        self.refresh()

    def change_waist_participation(self, checked):
        if self.runner or self.process or self.reserved:
            return
        self.sequence = Sequence()
        self.scene_confirm.setChecked(False)
        self.message = ('已允许腰部参与 IK；旧计划已清除，请重新预检。'
                        if checked else '已固定腰部；旧计划已清除，请重新预检。')
        self.refresh()

    @property
    def is_simulation(self):
        return self.execution_mode.currentData() == 'simulation'

    def change_execution_mode(self, *_):
        self.simulation = None
        self._sim_reserved = False
        self.sequence = Sequence()
        self.shape_override = None
        if self.simulation_display:
            self.simulation_display(None, {})
        self.message = '执行方式已切换，请重新预览；仿真是运动学播放，不模拟物体接触。'
        self.refresh()

    def simulation_backend(self):
        if self.simulation is None:
            if self.simulation_display is None:
                raise ValueError('未连接机器人页面仿真显示')
            from grasp_simulation import GraspSimulation
            profile = load_profile(self.profile_path.text().strip(), require_calibration=False)
            seed = (self._startup_snapshot or {}).get('q16', self._startup_seed)
            self.simulation = GraspSimulation(profile, seed, self.simulation_display)
            self.simulation.hands = copy.deepcopy((self._startup_snapshot or {}).get('hands', {'left':[0.]*6, 'right':[0.]*6}))
            self._simulation_startup = dict(self.simulation.feedback(), hands=copy.deepcopy(self.simulation.hands))
        return self.simulation

    def open_simulation_controls(self):
        if not self.is_simulation:
            return
        try:
            self._startup_recording = False
            simulation = self.simulation_backend()
            simulation.publish(simulation.q)
            self.message = '已打开机器人页面仿真手控制；仅改变模型，不发送真机命令。'
        except Exception as exc:
            self.message = '仿真控制不可用：' + str(exc)
        self.refresh()

    def save_plan_validity(self):
        if self.reserved or self.process or self.runner:
            return
        try:
            path = Path(self.profile_path.text().strip())
            data = json.loads(path.read_text())
            data['max_plan_age_s'] = self.plan_validity.value()
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
            self.sequence = Sequence()
            self.message = f'规划有效期已保存为 {self.plan_validity.value():g} 秒；请重新预览。'
        except Exception as exc:
            self.message = '有效期未保存：' + str(exc)
        self.refresh()

    def show_shape_result(self):
        self.suggest_shape()
        self.shape_result_dialog = QMessageBox(self)
        self.shape_result_dialog.setWindowTitle('物体手型估计结果')
        self.shape_result_dialog.setText(self.hand_shape_label.text())
        self.shape_result_dialog.setStandardButtons(QMessageBox.Ok)
        self.shape_result_dialog.setModal(False)
        self.shape_result_dialog.show()

    def suggest_shape(self):
        from object_hand_shape import calibrated_preset, suggest_hand_shape
        if self.reserved or self.process or self.runner:
            return
        self.sequence = Sequence()
        self.shape_override = None
        try:
            self.hand_shape_suggestion = suggest_hand_shape(self.observation)
            item = self.hand_shape_suggestion
            dims = ' × '.join(f'{x*1000:.0f}' for x in item['extents_m'])
            text = (f"建议：{item['name']}；参与手指：{'、'.join(item['fingers'])}\n"
                f"估计尺寸：{dims} mm；建议预张开约 {item['suggested_aperture_m']*1000:.0f} mm（需实测手型核对）\n"
                + item['note'] + '\n' + item['limitations'])
            try:
                if self.observation.get('execution_blockers'):
                    raise ValueError('仅供参考，不能应用到运动：' + '；'.join(self.observation['execution_blockers']))
                profile_path = self.profile_path.text().strip()
                profile = load_profile(profile_path, require_calibration=not self.is_simulation)
                opened, closed = calibrated_preset(profile, item)
                self.shape_override = (self.observation['observation_id'], profile_path, opened, closed)
                self.sequence = Sequence()
                text += (f"\n已用于本次抓取：预张开 {[round(x, 3) for x in opened]}；"
                         f"闭合 {[round(x, 3) for x in closed]}。请重新预览。")
                self.message = '已根据物体尺寸采用手型参数；没有发送手部或手臂命令，请重新预览。'
            except Exception as exc:
                self.shape_override = None
                text += '\n尚未应用到抓取：' + str(exc)
                self.message = '已生成手型建议；尚未应用：' + str(exc)
            self.hand_shape_label.setText(text)
        except Exception as exc:
            self.hand_shape_suggestion = None
            self.shape_override = None
            self.hand_shape_label.setText('手型建议不可用：'+str(exc))
            self.message = '手型估计失败：' + str(exc)
        self.refresh()

    def apply_shape(self):
        from object_hand_shape import calibrated_preset
        try:
            if self.reserved or self.process or not self.hand_shape_suggestion:
                raise ValueError('请先估计手型并结束当前动作')
            if self.observation.get('execution_blockers'):
                raise ValueError('；'.join(self.observation['execution_blockers']))
            profile = self.selected_profile(require_calibration=True)
            opened, closed = calibrated_preset(profile, self.hand_shape_suggestion)
            self.shape_override = (self.observation['observation_id'], self.profile_path.text().strip(), opened, closed)
            self.sequence = Sequence()
            self.message = '已选择该物体的标定手型；重新预览后生效，没有发送开合命令。'
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def selected_profile(self, require_calibration=True):
        profile = load_profile(self.profile_path.text().strip(), require_calibration=require_calibration and not self.is_simulation)
        side = str(self.grasp_arm.currentData() or profile['arm_side'])
        profile['arm_side'] = side
        profile['tcp_link'] = 'Link7_l' if side == 'left' else 'Link7_r'
        override = getattr(self, 'shape_override', None)
        if override and self.observation and override[:2] == (self.observation['observation_id'], self.profile_path.text().strip()):
            profile['open_positions'], profile['close_positions'] = override[2:]
        return profile

    def browse_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择已标定的抓取配置', str(self.root / 'config'), 'JSON (*.json)')
        if path:
            self.profile_path.setText(path)
            self.check_readiness()

    def go_perception(self):
        if self.open_perception:
            self.open_perception()
        else:
            self.message = '请打开「分割」页，选择彩色图并点击物体；此处不接受旧多视角面板的近似位姿。'
            self.refresh()

    def start_feedback(self):
        if self.ensure_feedback is None:
            self.message = '未配置 FK 启动入口，请在宿主机运行 scripts/start_host_fk.sh start。'
        else:
            self.ensure_feedback()
            self.message = '已请求启动 FK 反馈；等待 2 秒后点击“检查准备状态”。'
        self.refresh()

    def go_calibration(self):
        if self.open_calibration:
            self.open_calibration()
        else:
            self.message = '未配置抓取标定页。请填写 T_grasp_tcp、实测开合量及工作空间。'
            self.refresh()

    def _ensure_extrinsic(self):
        if self.fill_extrinsic and self.observation:
            self.observation = self.fill_extrinsic(self.observation)

    def check_readiness(self):
        lines = []
        profile = None
        try:
            path = Path(self.profile_path.text().strip())
            if not path.is_file():
                raise ValueError(f'抓取配置不存在：{path}。请先填写 config/grasp_execution.example.json 中的实际标定数据并另存；模板不能直接用于执行。')
            profile = self.selected_profile(require_calibration=False)
            lines.append(f"✓ 当前选择：{'左臂 / Link7_l' if profile['arm_side'] == 'left' else '右臂 / Link7_r'}")
            if profile.get('calibrated'):
                lines.append('✓ 抓取配置已通过格式和标定声明检查')
            else:
                lines.append('抓取配置格式可用，但尚未完成手型/TCP 标定：可以生成候选并预览，不能张开手或执行运动。')
            for key, label in (('planner_python', 'IK Python'), ('urdf_path', '机器人 URDF')):
                if not Path(profile[key]).is_file():
                    lines.append(f'缺少 {label}：{profile[key]}')
        except Exception as exc:
            lines.append(str(exc))
        self._ensure_extrinsic()
        if self.observation is None:
            lines.append('未收到视觉6D估计：点「前往视觉6D估计」，选择物体并点击「估计一次」。')
        else:
            if 'T_base_camera' not in self.observation:
                detail = self.observation.get('extrinsic_error') or '需要相机光学系到 base_link 的真实外参'
                lines.append(f'估计缺少 T_base_camera：{detail}。')
            if not self.observation.get('grasps', {}).get('candidates'):
                lines.append('尚无抓取候选：外参和实时 TCP 就绪后，点「按当前手朝向生成候选」。')
        try:
            self.feedback()
            lines.append('✓ 仿真状态就绪（初始读取当前关节，缺少反馈则用零位）' if self.is_simulation else '✓ 实测关节反馈新鲜')
        except Exception as exc:
            lines.append('反馈未就绪：'+str(exc))
            lines.append('可点“启动 FK 反馈”；它只订阅关节并发布 FK 状态，不发送手臂命令。')
        if not self.is_simulation and not self.node.is_arm_enabled():
            lines.append('手臂未使能：执行前到「手动控制」启用手臂。')
        if not self.is_simulation and not self.node.is_hand_enabled():
            lines.append('手未使能：执行前到「手动控制」启用手。')
        self.readiness_label.setText('\n'.join(lines))

    def invalidate_preview(self, *_):
        if not self.reserved and self.process is None:
            self.sequence = Sequence()
            self.refresh()

    def offset_changed(self, *_):
        if not self.reserved and self.observation and self.observation.get('grasps', {}).get('model') == 'manual_current_hand_orientation':
            self.observation.pop('grasps', None)
            self.candidates.clear()
            self.sequence = Sequence()
            self.message = '接触偏移已改变，请重新生成候选并预览'
            self.refresh()

    def feedback(self):
        if self.is_simulation:
            return self.simulation_backend().feedback()
        return read_feedback(os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))

    def set_observation(self, observation):
        if self.reserved or self.process is not None:
            self.message = '抓取进行中：新观测未替换已锁定目标；需要更换时先停止'
            self.refresh()
            return
        self.observation = {k: v for k, v in observation.items() if k not in ('mask', 'image', 'K')}
        self._ensure_extrinsic()
        self.sequence = Sequence()
        self.scene_confirm.setChecked(False)
        self.populate_candidates()
        self.shape_override = None
        self.suggest_shape()
        self.refresh()

    def set_target_observations(self, targets):
        if self.reserved or self.process or self.runner:
            return
        selected = str(self.grasp_target_combo.currentData() or '')
        self.target_observations = copy.deepcopy(targets)
        self.grasp_target_combo.blockSignals(True)
        self.grasp_target_combo.clear()
        for target_id, entry in self.target_observations.items():
            self.grasp_target_combo.addItem(entry.get('label') or target_id, target_id)
        index = self.grasp_target_combo.findData(selected)
        self.grasp_target_combo.setCurrentIndex(index if index >= 0 else 0)
        self.grasp_target_combo.blockSignals(False)
        self._refresh_task_role_targets()
        if self.target_observations:
            self._on_grasp_target_changed()
        else:
            self.method_observations = {}
            self.observation = None
            self.sequence = Sequence()
            self.candidates.clear()
            self.shape_override = None
            self.refresh()

    def select_target_observation(self, target_id):
        index = self.grasp_target_combo.findData(str(target_id))
        if index >= 0:
            self.grasp_target_combo.setCurrentIndex(index)
            self._on_grasp_target_changed()
            return True
        return False

    def _on_grasp_target_changed(self, *_):
        if self.reserved or self.process or self.runner:
            return
        target_id = str(self.grasp_target_combo.currentData() or '')
        entry = self.target_observations.get(target_id)
        if entry:
            self.set_method_observations(entry.get('methods') or {})

    def change_task_workflow(self, *_):
        task_key = str(self.task_combo.currentData() or 'phone_box_assembly')
        saved = None
        if self.load_task_data:
            try:
                saved = self.load_task_data(task_key)
                if (saved and saved.get('task_definition')
                        and not TASKS.get(task_key, {}).get('builtin')):
                    TASKS[task_key] = validate_task(saved['task_definition'])
            except Exception as exc:
                self.task_data_status.setText('任务数据加载失败：' + str(exc))
        task = TASKS[task_key]
        self._render_task_controls(task)
        self.reset_task_workflow()
        if saved:
            self.set_target_observations(saved.get('target_observations') or {})
            state = saved.get('panel_state') or {}
            self._workflow_step_settings = copy.deepcopy(state.get('step_settings') or {})
            for role, target_id in (state.get('roles') or {}).items():
                combo = self.task_role_combos.get(role)
                if combo is not None:
                    index = combo.findData(target_id)
                    if index >= 0:
                        combo.setCurrentIndex(index)
            target_id = str(state.get('selected_target') or '')
            if target_id:
                self.select_target_observation(target_id)
            self._workflow_step = max(0, min(int(state.get('workflow_step', 0)), len(task['steps'])-1))
            actions = task['steps'][self._workflow_step].get('actions') or ()
            self._workflow_substep = max(0, min(int(state.get('workflow_substep', 0)),
                                                max(0, len(actions)-1)))
            self._workflow_finished = bool(state.get('workflow_finished', False))
            self.task_flow.set_current(self._workflow_step)
            if self._workflow_finished:
                self.task_flow.set_current(len(task['steps']))
            self._apply_workflow_step_settings(self._workflow_step, self._workflow_substep)
            saved_at = str(saved.get('saved_at_text') or '')
            self.task_data_status.setText('任务数据：已加载' + (f'（{saved_at}）' if saved_at else ''))
        elif self.load_task_data:
            self.set_target_observations({})
            self._workflow_step_settings = {}
            self.task_data_status.setText('任务数据：该任务尚无保存结果')

    def _render_task_controls(self, task):
        self.task_flow.set_task(task)
        while self.task_role_row.count():
            item = self.task_role_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.task_role_combos = {}
        for role, label in task['roles']:
            self.task_role_row.addWidget(QLabel(label+'目标'))
            combo = QComboBox()
            self.task_role_combos[role] = combo
            self.task_role_row.addWidget(combo)
        self.task_role_row.addStretch(1)
        self._refresh_task_role_targets()

    def refresh_task_selector(self, selected_key=''):
        selected_key = selected_key or str(self.task_combo.currentData() or '')
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        for key, task in TASKS.items():
            self.task_combo.addItem(task['name'], key)
        index = self.task_combo.findData(selected_key)
        self.task_combo.setCurrentIndex(index if index >= 0 else 0)
        self.task_combo.blockSignals(False)
        self.change_task_workflow()

    def _create_task_from_dialog(self, source=None):
        dialog = TaskEditorDialog(source, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            definition = dialog.definition()
            key = new_task_key()
            TASKS[key] = definition
            save_tasks(TASKS)
            self.refresh_task_selector(key)
            self.task_data_status.setText('任务定义已创建；运行数据尚未保存')
        except Exception as exc:
            self.task_data_status.setText('任务定义创建失败：' + str(exc))
            QMessageBox.warning(self, '任务未创建', str(exc))

    def new_task(self):
        self._create_task_from_dialog(None)

    def copy_task(self):
        source = TASKS.get(str(self.task_combo.currentData() or ''))
        if not source:
            return
        name, accepted = QInputDialog.getText(
            self, '复制任务', '副本名称', text=source['name']+' 副本')
        if not accepted:
            return
        try:
            name = name.strip()
            if not name:
                raise ValueError('副本名称不能为空')
            definition = validate_task(dict(copy.deepcopy(source), name=name, builtin=False))
            key = new_task_key()
            TASKS[key] = definition
            save_tasks(TASKS)
            self.refresh_task_selector(key)
            self.task_data_status.setText('任务定义复制成功；运行数据尚未保存')
            self.message = f'已复制任务“{source["name"]}”为“{name}”'
            self.refresh()
        except Exception as exc:
            self.task_data_status.setText('任务复制失败：' + str(exc))
            QMessageBox.warning(self, '任务复制失败', str(exc))

    def manage_tasks(self):
        action, ok = QInputDialog.getItem(
            self, '管理任务', '操作', ('重命名当前任务', '删除当前任务', '导出当前任务', '导入任务'), 0, False)
        if not ok:
            return
        key = str(self.task_combo.currentData() or '')
        task = TASKS.get(key)
        try:
            if action == '重命名当前任务':
                name, accepted = QInputDialog.getText(self, '重命名任务', '新名称', text=task['name'])
                if accepted and name.strip():
                    TASKS[key] = validate_task(dict(task, name=name.strip()))
                    save_tasks(TASKS)
                    self.refresh_task_selector(key)
            elif action == '删除当前任务':
                if task.get('builtin'):
                    raise ValueError('内置任务不能删除，可以复制后修改')
                if QMessageBox.question(self, '删除任务', f"删除任务定义“{task['name']}”？\n已保存运行数据不会删除。") == QMessageBox.Yes:
                    TASKS.pop(key)
                    save_tasks(TASKS)
                    self.refresh_task_selector()
            elif action == '导出当前任务':
                path, _ = QFileDialog.getSaveFileName(self, '导出任务', task['name']+'.json', 'JSON (*.json)')
                if path:
                    Path(path).write_text(json.dumps({'version':1,'task':task}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            else:
                path, _ = QFileDialog.getOpenFileName(self, '导入任务', '', 'JSON (*.json)')
                if path:
                    data = json.loads(Path(path).read_text(encoding='utf-8'))
                    imported = validate_task(data.get('task') or data)
                    imported['builtin'] = False
                    new_key = new_task_key()
                    TASKS[new_key] = imported
                    save_tasks(TASKS)
                    self.refresh_task_selector(new_key)
        except Exception as exc:
            QMessageBox.warning(self, '任务管理失败', str(exc))

    def current_task_state(self):
        self._capture_workflow_step_settings(self._workflow_step, self._workflow_substep)
        return {
            'roles': {role: str(combo.currentData() or '') for role, combo in self.task_role_combos.items()},
            'selected_target': str(self.grasp_target_combo.currentData() or ''),
            'workflow_step': self._workflow_step,
            'workflow_substep': self._workflow_substep,
            'workflow_finished': self._workflow_finished,
            'arm_side': str(self.grasp_arm.currentData() or ''),
            'allow_waist': self.waist_participation.isChecked(),
            'motion_stage': str(self.motion_stage.currentData() or ''),
            'pose_method': str(self.source_combo.currentData() or ''),
            'step_settings': copy.deepcopy(self._workflow_step_settings),
        }

    def _workflow_action(self, index=None, substep=None):
        task = TASKS.get(str(self.task_combo.currentData() or ''))
        index = self._workflow_step if index is None else int(index)
        substep = self._workflow_substep if substep is None else int(substep)
        if not task or not 0 <= int(index) < len(task['steps']):
            return None
        step = task['steps'][index]
        if step['kind'] == 'sequence':
            actions = step.get('actions') or ()
            return actions[substep] if 0 <= substep < len(actions) else None
        return step

    def _workflow_setting_key(self, index, substep):
        return f'{int(index)}:{int(substep)}'

    def _capture_workflow_step_settings(self, index, substep=0):
        step = self._workflow_action(index, substep)
        if step is None:
            return
        role = str(step.get('role') or '')
        role_combo = self.task_role_combos.get(role)
        self._workflow_step_settings[self._workflow_setting_key(index, substep)] = {
            'step': copy.deepcopy(step),
            'target_id': str(role_combo.currentData() or '') if role_combo else '',
            'arm_side': str(self.grasp_arm.currentData() or ''),
            'allow_waist': self.waist_participation.isChecked(),
            'pose_method': str(self.source_combo.currentData() or ''),
            'execution_mode': str(self.execution_mode.currentData() or ''),
            'motion_stage': str(self.motion_stage.currentData() or ''),
            'contact_offsets_cm': [spin.value() for spin in self.contact_offsets],
            'wrench_guard': self.wrench_guard_check.isChecked(),
            'force_limit_n': self.force_limit.value(),
            'torque_limit_nm': self.torque_limit.value(),
        }

    def _apply_workflow_step_settings(self, index, substep=0):
        settings = (self._workflow_step_settings.get(self._workflow_setting_key(index, substep))
                    or self._workflow_step_settings.get(str(int(index))))
        if not settings:
            return
        execution = self.execution_mode.findData(settings.get('execution_mode'))
        if execution >= 0 and not self._workflow_auto_running:
            self.execution_mode.setCurrentIndex(execution)
        arm = self.grasp_arm.findData(settings.get('arm_side'))
        if arm >= 0:
            self.grasp_arm.setCurrentIndex(arm)
        self.waist_participation.setChecked(bool(settings.get('allow_waist', False)))
        method = self.source_combo.findData(settings.get('pose_method'))
        if method >= 0 and self.source_combo.model().item(method).isEnabled():
            self.source_combo.setCurrentIndex(method)
        stage = self.motion_stage.findData(settings.get('motion_stage'))
        if stage >= 0:
            self.motion_stage.setCurrentIndex(stage)
        for spin, value in zip(self.contact_offsets, settings.get('contact_offsets_cm') or []):
            spin.setValue(float(value))
        self.wrench_guard_check.setChecked(bool(settings.get('wrench_guard', True)))
        self.force_limit.setValue(float(settings.get('force_limit_n', self.force_limit.value())))
        self.torque_limit.setValue(float(settings.get('torque_limit_nm', self.torque_limit.value())))
        target_id = str(settings.get('target_id') or '')
        step = self._workflow_action(index, substep) or {}
        role_combo = self.task_role_combos.get(str(step.get('role') or ''))
        if role_combo is not None and target_id:
            target_index = role_combo.findData(target_id)
            if target_index >= 0:
                role_combo.setCurrentIndex(target_index)

    def save_current_task(self):
        if self.runner or self.process or self.reserved:
            self.message = '请先停止当前任务，再保存任务数据。'
            self.refresh()
            return
        if not self.save_task_data:
            self.task_data_status.setText('任务数据保存不可用：未连接保存入口')
            return
        try:
            result = self.save_task_data(str(self.task_combo.currentData()), self.current_task_state())
            self.task_data_status.setText('任务数据：' + str(result or '保存完成'))
            self.message = str(result or '任务数据保存完成')
        except Exception as exc:
            self.task_data_status.setText('任务数据保存失败：' + str(exc))
            self.message = '任务数据保存失败：' + str(exc)
        self.refresh()

    def _refresh_task_role_targets(self):
        if not hasattr(self, 'task_role_combos'):
            return
        for combo in self.task_role_combos.values():
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem('请选择…', '')
            for target_id, entry in self.target_observations.items():
                combo.addItem(entry.get('label') or target_id, target_id)
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def reset_task_workflow(self):
        if self.runner or self.process:
            return
        self._workflow_step = 0
        self._workflow_substep = 0
        self._workflow_finished = False
        self._workflow_waiting = False
        self.task_flow.set_current(0)
        self.message = f"{self.task_combo.currentText()}：请选择任务目标，然后执行流程下一步。"
        self.refresh()

    def select_workflow_step(self, index):
        if self.runner or self.process or self._workflow_waiting or self._workflow_auto_running:
            self.message = '当前正在规划或执行，不能跳转流程步骤。'
            self.refresh()
            return
        task = TASKS[str(self.task_combo.currentData())]
        index = max(0, min(int(index), len(task['steps'])-1))
        self._workflow_step = index
        self._workflow_substep = 0
        self._workflow_finished = False
        self.sequence = Sequence()
        self.scene_confirm.setChecked(False)
        self.task_flow.set_current(index)
        self._apply_workflow_step_settings(index, 0)
        action = self._workflow_action()
        role = str((action or {}).get('role') or '')
        combo = self.task_role_combos.get(role)
        target_id = str(combo.currentData() or '') if combo else ''
        if target_id:
            self.select_target_observation(target_id)
        top = task['steps'][index]
        self.message = f"已跳转到：{top['label']}" + (
            f" → {action['label']}" if top['kind'] == 'sequence' and action else '')
        self.refresh()

    def _advance_task_workflow(self):
        task = TASKS[str(self.task_combo.currentData())]
        completed_action = self._workflow_action()
        completed_kind = (completed_action or {}).get('kind', 'confirm')
        top = task['steps'][self._workflow_step]
        actions = top.get('actions') or ()
        if actions and self._workflow_substep+1 < len(actions):
            self._workflow_substep += 1
            self.task_flow.set_current(self._workflow_step)
            self._apply_workflow_step_settings(self._workflow_step, self._workflow_substep)
            action = self._workflow_action()
            self.message = f"流程：{top['label']} → {action['label']}"
        elif self._workflow_step+1 < len(task['steps']):
            self._workflow_step += 1
            self._workflow_substep = 0
            self.task_flow.set_current(self._workflow_step)
            self._apply_workflow_step_settings(self._workflow_step, 0)
            step = task['steps'][self._workflow_step]
            action = self._workflow_action()
            self.message = f"流程进入：{step['label']}" + (f" → {action['label']}" if step['kind']=='sequence' else '')
        else:
            self._workflow_finished = True
            self.task_flow.set_current(len(task['steps']))
            self.message = f"{task['name']}流程已完成。"
        if (self._workflow_auto_running and not self._workflow_auto_cancel_requested
                and not self._workflow_finished):
            self._schedule_auto_workflow(1000 if completed_kind == 'hand' else 200)

    def _workflow_target(self, role):
        combo = self.task_role_combos.get(role)
        target_id = str(combo.currentData() or '') if combo else ''
        if not target_id or target_id not in self.target_observations:
            raise ValueError(f"请先选择{dict(TASKS[str(self.task_combo.currentData())]['roles'])[role]}目标")
        for other_role, other_combo in self.task_role_combos.items():
            other_id = str(other_combo.currentData() or '')
            if other_role != role and other_id and target_id == other_id:
                labels = dict(TASKS[str(self.task_combo.currentData())]['roles'])
                raise ValueError(f"{labels[role]}目标和{labels[other_role]}目标不能相同")
        self.select_target_observation(target_id)
        return target_id

    def _workflow_hand(self, opened):
        profile = self.selected_profile()
        self.reserve(require_preview=False)
        values = profile['open_positions' if opened else 'close_positions']
        if self.is_simulation:
            self.simulation_backend().hand(profile['arm_side'], values)
        else:
            self.node._grasp_hand_command_authorized = True
            try:
                self.node.apply_hand_joint_positions(profile['arm_side'], values)
            finally:
                self.node._grasp_hand_command_authorized = False
        self.set_reserved(False)

    def execute_workflow_next(self):
        if self.runner or self.process or self._workflow_waiting:
            return
        task = TASKS[str(self.task_combo.currentData())]
        step = self._workflow_action()
        if step is None or self._workflow_finished:
            self.message = f"{task['name']}流程已完成。"
            self.refresh()
            return
        try:
            self._capture_workflow_step_settings(self._workflow_step, self._workflow_substep)
            kind = step['kind']
            if kind == 'target':
                self._workflow_target(step['role'])
                self._advance_task_workflow()
            elif kind == 'motion':
                self._workflow_target(step['role'])
                self.motion_stage.setCurrentIndex(self.motion_stage.findData(step['stage']))
                self._workflow_waiting = True
                self.execute()
                if self.runner is None and self.process is None:
                    self._workflow_waiting = False
            elif kind == 'hand':
                self._workflow_target(step['role'])
                self._workflow_hand(step['open'])
                self._advance_task_workflow()
            elif kind == 'confirm':
                self._advance_task_workflow()
            elif kind == 'complete':
                self._advance_task_workflow()
        except Exception as exc:
            self._workflow_waiting = False
            self.set_reserved(False)
            self.message = str(exc)
            if self._workflow_auto_running:
                self._workflow_auto_paused = True
        self.refresh()

    def _capture_auto_workflow_start(self):
        if self.is_simulation:
            simulation = self.simulation_backend()
            return dict(simulation.feedback(), hands=copy.deepcopy(simulation.hands))
        measured = copy.deepcopy(self.feedback())
        try:
            measured['hands'] = self.read_startup_hands()
        except Exception:
            measured['hands'] = {}
        return measured

    def toggle_all_workflow(self):
        if self._workflow_auto_running:
            self.cancel_all_workflow()
            return
        if self.runner or self.process or self.reserved:
            self.message = '请先停止当前动作，再执行全部流程。'
            self.refresh()
            return
        try:
            for role, label in TASKS[str(self.task_combo.currentData())]['roles']:
                combo = self.task_role_combos.get(role)
                if combo is None or not combo.currentData():
                    raise ValueError(f'请先选择{label}目标')
            self._workflow_step = 0
            self._workflow_substep = 0
            self._workflow_finished = False
            self._workflow_waiting = False
            self.task_flow.set_current(0)
            self._apply_workflow_step_settings(0, 0)
            if not self.is_simulation and not self.scene_confirm.isChecked():
                raise ValueError('真机全部执行前，请确认完整流程路径及手周围无障碍')
            self._workflow_auto_initial_snapshot = self._capture_auto_workflow_start()
            self._workflow_auto_running = True
            self._workflow_auto_paused = False
            self._workflow_auto_cancel_requested = False
            self._workflow_auto_restoring = False
            self.message = f'{self.task_combo.currentText()}：开始全部执行。'
            self._schedule_auto_workflow(0)
        except Exception as exc:
            self.message = '全部执行未启动：' + str(exc)
        self.refresh()

    def _schedule_auto_workflow(self, delay_ms=250):
        if (self._workflow_auto_running and not self._workflow_auto_paused
                and not self._workflow_auto_cancel_requested):
            self._workflow_auto_timer.start(max(0, int(delay_ms)))

    def _run_auto_workflow_next(self):
        if (not self._workflow_auto_running or self._workflow_auto_paused
                or self._workflow_auto_cancel_requested or self.runner or self.process
                or self._workflow_waiting):
            return
        task = TASKS[str(self.task_combo.currentData())]
        step = self._workflow_action()
        if self._workflow_finished or step is None:
            self._workflow_auto_running = False
            self._workflow_auto_initial_snapshot = None
            self._workflow_auto_paused = False
            self.message = f'{task["name"]}：全部流程执行完成。'
            self.refresh()
            return
        self.execute_workflow_next()
        if self._workflow_finished:
            self._workflow_auto_running = False
            self._workflow_auto_initial_snapshot = None
            self._workflow_auto_paused = False
            self.message = f'{task["name"]}：全部流程执行完成。'
            self.refresh()

    def toggle_auto_workflow_pause(self):
        if not self._workflow_auto_running or self._workflow_auto_cancel_requested:
            return
        if self._workflow_auto_paused:
            if self.runner is not None and self.runner.state == 'paused':
                self.pause_or_resume()
                if self.runner.state != 'running':
                    return
            elif self.runner is None and self.process is None and self.sequence.state == 'planned':
                self._execute_ready_plan()
                if self.runner is None:
                    return
            self._workflow_auto_paused = False
            self.message = '全部流程继续执行。'
            self._schedule_auto_workflow(0)
        else:
            self._workflow_auto_paused = True
            self._workflow_auto_timer.stop()
            if self.runner is not None and self.runner.state == 'running':
                self.pause_or_resume()
            self.message = '全部流程已暂停；当前轨迹保持位置。'
        self.refresh()

    def cancel_all_workflow(self):
        if not self._workflow_auto_running:
            return
        self._workflow_auto_cancel_requested = True
        self._workflow_auto_paused = False
        self._workflow_auto_timer.stop()
        self._execute_after_plan = False
        self._workflow_waiting = False
        if self.process is not None:
            self.sequence.stop(motion_active=False)
            self.process.terminate()
        if self.runner is not None:
            self.runner.stop('全部流程取消，准备恢复流程初始状态')
        self.message = '正在取消全部流程；停止确认后恢复第一个流程的初始状态。'
        if self.process is None and self.runner is None:
            self._begin_auto_workflow_restore()
        self.refresh()

    def _begin_auto_workflow_restore(self):
        if self._workflow_auto_restoring or not self._workflow_auto_initial_snapshot:
            return
        self._workflow_auto_restoring = True
        try:
            self._start_restore_snapshot(self._workflow_auto_initial_snapshot, auto_restore=True)
            if self.runner is None:
                raise ValueError(self.message or '恢复轨迹未启动')
            self.message = '全部流程已取消，正在恢复流程初始双臂和双手状态。'
        except Exception as exc:
            self._finish_auto_workflow_cancel(False, str(exc))

    def _finish_auto_workflow_cancel(self, restored, detail=''):
        self._workflow_auto_timer.stop()
        self._workflow_auto_running = False
        self._workflow_auto_paused = False
        self._workflow_auto_cancel_requested = False
        self._workflow_auto_restoring = False
        self._workflow_auto_initial_snapshot = None
        self._workflow_waiting = False
        self._workflow_step = 0
        self._workflow_substep = 0
        self._workflow_finished = False
        self.task_flow.set_current(0)
        self._apply_workflow_step_settings(0, 0)
        self.message = ('全部流程已取消并恢复到流程初始状态。' if restored else
                        '全部流程已取消，但恢复初始状态失败：' + detail)
        self.refresh()

    def set_method_observations(self, observations):
        if self.reserved or self.process or self.runner:
            return
        selected = self.source_combo.currentData()
        self.method_observations = copy.deepcopy(observations)
        self.source_combo.blockSignals(True)
        for i, label in enumerate(('PCA', 'FoundationPose', 'Any6D')):
            key = self.source_combo.itemData(i)
            available = key in observations
            self.source_combo.setItemText(i, label if available else label + '（本轮无结果）')
            self.source_combo.model().item(i).setEnabled(available)
        if selected not in observations:
            selected = next(iter(observations), None)
        self.source_combo.setCurrentIndex(self.source_combo.findData(selected))
        self.source_combo.blockSignals(False)
        self.select_observation_source()

    def select_observation_source(self, *_):
        if self.reserved or self.process or self.runner:
            return
        observation = self.method_observations.get(self.source_combo.currentData())
        if observation is None:
            self.observation = None
            self.shape_override = None
            self.hand_shape_suggestion = None
            self.sequence = Sequence()
            self.candidates.clear()
            self.message = '本轮没有可用的方法结果，请先进行视觉6D估计。'
            self.refresh()
            return
        self.set_observation(copy.deepcopy(observation))
        self.check_readiness()

    @property
    def reserved(self):
        if self.is_simulation:
            return self._sim_reserved
        return bool(getattr(self.node, '_grasp_execution_reserved', False))

    def set_reserved(self, value):
        if self.is_simulation:
            self._sim_reserved = value
        else:
            self.node._grasp_execution_reserved = value

    def populate_candidates(self):
        self.candidates.clear()
        for i, candidate in enumerate(self.observation.get('grasps', {}).get('candidates', [])):
            label = (f"人工候选 {i+1}" if candidate.get('source') == 'manual'
                     else f"候选 {i+1} / {candidate.get('model_score', 0):.3f}")
            self.candidates.addItem(label, i)

    def make_manual_candidate(self):
        try:
            if self.reserved or self.process or not self.observation:
                raise ValueError('请先结束当前任务并获取点击观测')
            self._ensure_extrinsic()
            profile = self.selected_profile(require_calibration=False)
            self.observation['grasps'] = manual_candidate(
                self.observation, profile, self.feedback(), [s.value()/100 for s in self.contact_offsets])
            self.sequence = Sequence()
            self.populate_candidates()
            extra = '' if profile.get('calibrated') else ' 配置未标定，只能预览，不能执行。'
            self.message = '已生成候选：人工接触点+当前手朝向。请预览并核对接触位置，尚未做自动避障。' + extra
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def import_candidates(self):
        if not self.observation or self.reserved or self.process:
            return
        path, _ = QFileDialog.getOpenFileName(self, '导入本次观测的手型抓取候选', '', 'JSON (*.json)')
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            if data.get('observation_id') != self.observation.get('observation_id'):
                raise ValueError('候选不属于当前观测，不能复用旧坐标')
            if not data.get('ok') or not data.get('candidates') or not data.get('gripper'):
                raise ValueError('候选文件需要 ok、gripper 和 candidates')
            from click_pose import validate_transform
            for candidate in data['candidates']:
                validate_transform(candidate['T_camera_grasp'])
                if not np.isfinite(float(candidate.get('model_score', 0))):
                    raise ValueError('候选分数无效')
            self.observation['grasps'] = data
            self.sequence = Sequence()
            self.populate_candidates()
            self.message = '候选已导入，仍需核对实际手型并重新预览'
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def preview(self):
        try:
            if self.reserved or self.process:
                raise ValueError('已有任务，请先停止')
            if not self.observation:
                raise ValueError('请先点击估计')
            profile = self.selected_profile(require_calibration=False)
            self._ensure_extrinsic()
            if not self.observation.get('grasps', {}).get('candidates'):
                self.observation['grasps'] = manual_candidate(
                    self.observation, profile, self.feedback(), [s.value()/100 for s in self.contact_offsets])
                self.populate_candidates()
            result = make_preview(self.observation, self.candidates.currentIndex(), profile)
            self.sequence.reset(result, profile)
            if self.robot_preview_callback:
                self.robot_preview_callback(result)
            if self.is_simulation:
                self._startup_recording = False
                simulation = self.simulation_backend()
                simulation.publish(simulation.q)
            if self.preview_image_callback is not None:
                pixmap = self.preview_image_callback(result)
                if pixmap.isNull():
                    self.preview_image.setText('使用手动控制里的融合6D结果；图像叠加请在来源页查看，下方为机器人坐标目标。')
                else:
                    self.preview_image.setPixmap(pixmap)
            extra = '' if profile.get('calibrated') else ' 当前 T_grasp_tcp 未标定，预览仅供核对，不能执行。'
            self.message = '预览完成；可独立验证手型或执行所选阶段，尚未验证机械臂轨迹。' + extra
        except Exception as exc:
            self.sequence.preview = None
            self.message = str(exc)
            if self.robot_preview_callback:
                self.robot_preview_callback(getattr(exc, 'preview', None))
        self.refresh()

    def reserve(self, require_preview=True):
        if not self.is_simulation and self.other_busy():
            raise ValueError('已有手臂移动/文本队列/遥控，请先结束，不能同时控制')
        if (not self.scene_confirm.isChecked() and not self.is_simulation
                and not self._workflow_auto_restoring):
            raise ValueError('本阶段尚未接入自动避障，请先确认本段路径周围无障碍')
        if not self.is_simulation and (not self.node.is_arm_enabled() or not self.node.is_hand_enabled()):
            raise ValueError('请先手动启用手臂和手；抓取流程不自动使能')
        if require_preview and not self.sequence.preview:
            raise ValueError('请先预览抓取位置')
        profile = self.sequence.profile if require_preview else self.selected_profile()
        if not self.is_simulation and not profile.get('calibrated'):
            raise ValueError('抓取配置尚未完成手型/TCP 标定，不能发送开合或手臂运动')
        self.set_reserved(True)

    def hand(self):
        try:
            self._startup_recording = False
            if self.runner or self.process:
                raise ValueError('请先结束当前轨迹或预检')
            self.reserve(require_preview=False)
            stage = 'close' if self.sequence.stage == 'close' else 'open'
            self.feedback()
            profile = self.selected_profile()
            values = profile['open_positions' if stage == 'open' else 'close_positions']
            if self.is_simulation:
                self.simulation_backend().hand(profile['arm_side'], values)
            else:
                self.node._grasp_hand_command_authorized = True
                try:
                    self.node.apply_hand_joint_positions(profile['arm_side'], values)
                finally:
                    self.node._grasp_hand_command_authorized = False
            if self.sequence.preview and self.sequence.stage in ('open', 'close'):
                self.sequence.hand_sent()
            self.message = '手部开合验证已发送；不会自动执行手臂。'
        except Exception as exc:
            if self.sequence.state == 'planning' and self.process is None:
                self.sequence.fail(str(exc))
            self.message = str(exc)
        finally:
            if not self.runner and not self.process and self.sequence.state != 'stop_unconfirmed':
                self.set_reserved(False)
        self.refresh()

    def confirm(self):
        try:
            if self.sequence.state in ('await_open', 'await_grasp'):
                self.sequence.confirm_hand()
            self.message = '已记录手部人工确认；该确认不作为执行本段的前提，不会发送运动命令。'
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def plan(self, execute_after=False):
        try:
            if self.runner or self.process or self.sequence.state == 'stop_unconfirmed':
                raise ValueError('请先结束当前任务或确认停止状态')
            self._execute_after_plan = bool(execute_after)
            self.sequence = Sequence()
            profile = self.selected_profile(require_calibration=False)
            if not self.observation:
                raise ValueError('请先获取视觉估计结果')
            self._ensure_extrinsic()
            if not self.observation.get('grasps', {}).get('candidates'):
                self.observation['grasps'] = manual_candidate(
                    self.observation, profile, self.feedback(), [s.value()/100 for s in self.contact_offsets])
                self.populate_candidates()
            result = make_preview(self.observation, self.candidates.currentIndex(), profile)
            self.sequence.reset(result, profile)
            if execute_after:
                self.reserve()
            else:
                self.set_reserved(True)
            stage = self.sequence.begin_plan(self.motion_stage.currentData())
            feedback = self.feedback()
            self.launch('plan', stage=stage, start_joints=feedback['q16'],
                        allow_waist=self.waist_participation.isChecked())
        except Exception as exc:
            self._execute_after_plan = False
            if self.process is None and self.runner is None and self.sequence.state != 'stop_unconfirmed':
                self.sequence.fail(str(exc))
                self.set_reserved(False)
            self.message = str(exc)
        self.refresh()

    def execute(self):
        if self.runner or self.process or self.sequence.state == 'stop_unconfirmed':
            return
        plan = self.sequence.plan
        valid = (self.sequence.state == 'planned' and plan is not None
                 and plan.get('stage') == self.motion_stage.currentData()
                 and time.monotonic()-plan['accepted_at'] <= self.sequence.profile['max_plan_age_s'])
        if not valid:
            self.message = '执行请求已收到，正在自动计算目标并预检 IK；通过后才会启动运动。'
            self.execution_feedback.setText(self.message)
            self.plan(execute_after=True)
            return
        self._execute_ready_plan()

    def _execute_ready_plan(self):
        try:
            self._startup_recording = False
            self.reserve()
            if not self.is_simulation:
                sides = (('left', 'right') if (self.sequence.plan.get('arm_side') == 'both'
                                               or self.sequence.plan.get('other_tcp_held'))
                         else (self.sequence.plan['arm_side'],))
                violation = self.wrench_guard_violation(sides)
                if violation:
                    raise ValueError('六维力保护未就绪：' + violation)
            plan = self.sequence.begin_execute(
                manual_path_confirmed=(self.scene_confirm.isChecked() or self.is_simulation
                                       or self._workflow_auto_restoring)
            )
            if self.is_simulation:
                simulation = self.simulation_backend()
                self.runner = JointRunner(plan, simulation.publish, simulation.feedback, lambda: True)
                simulation.publish(simulation.q)
                self.message = '仿真执行中：仅机器人页面播放，不发送真机命令。'
                self.refresh()
                return
            def publish(q):
                self.node._grasp_joint_command_authorized = True
                try:
                    self.node._publish_wbc_joint_target(q)
                finally:
                    self.node._grasp_joint_command_authorized = False
            self.runner = JointRunner(plan, publish, self.feedback,
                                      lambda: self.node.is_arm_enabled() and self.node.is_hand_enabled())
            self.node._grasp_mode_command_authorized = True
            try:
                self.node._enter_model_mode_for_motion()
                self.node._release_ik_for_joint_move()
            finally:
                self.node._grasp_mode_command_authorized = False
            self.message = '正在执行本段；只发送预检过的关节路径并持续核对实测反馈'
        except Exception as exc:
            if self.runner is not None:
                self.runner.stop(str(exc), failed=True)
            elif self.sequence.state == 'executing':
                self.sequence.fail(str(exc))
            else:
                self.set_reserved(False)
            self.message = str(exc)
        self.refresh()

    def pause_or_resume(self):
        if self.runner is None:
            return
        try:
            if self.runner.state == 'paused':
                if self._wrench_guard_triggered:
                    side = self.runner.plan.get('arm_side', 'right')
                    sides = (('left', 'right') if (side == 'both' or self.runner.plan.get('other_tcp_held'))
                             else (side,))
                    violation = self.wrench_guard_violation(sides)
                    if violation:
                        self.message = '六维力仍不满足继续条件：' + violation
                        self.refresh()
                        return
                    self._wrench_guard_triggered = None
                self.runner.resume()
            else:
                self.runner.pause()
            self.message = self.runner.message
        except Exception as exc:
            self.runner.stop(str(exc), failed=True)
            self.message = str(exc)
        self.refresh()

    def launch(self, mode, **extra):
        if self.process is not None:
            raise ValueError('已有规划/执行请求')
        self.mode = mode
        self.job_directory = self.root / '.cache' / 'grasp_execution' / uuid.uuid4().hex
        self.job_directory.mkdir(parents=True)
        request = dict(mode=mode, execution_mode=('simulation' if self.is_simulation else 'hardware'),
                       profile=self.sequence.profile, preview=self.sequence.preview, **extra)
        (self.job_directory / 'request.json').write_text(json.dumps(request, ensure_ascii=False, allow_nan=False))
        self.process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        prefix = Path(self.sequence.profile['planner_python']).absolute().parent.parent
        for cmeel in sorted(prefix.glob('lib/python*/site-packages/cmeel.prefix')):
            environment.insert('LD_LIBRARY_PATH', str(cmeel / 'lib') + ':' + environment.value('LD_LIBRARY_PATH'))
            for packages in sorted(cmeel.glob('lib/python*/site-packages')):
                environment.insert('PYTHONPATH', str(packages) + ':' + environment.value('PYTHONPATH'))
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.log_tail = ''
        self.started = time.monotonic()
        self.process.start(self.sequence.profile['planner_python'], [str(self.root / 'host_grasp_plan_worker.py'),
                           '--request', str(self.job_directory / 'request.json'),
                           '--output', str(self.job_directory / 'result.json')])
        self.message = '宿主机 IK 预检中（不会发送运动命令）…'
        self.execution_feedback.setText(self.message)

    def read_output(self):
        if self.process:
            text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
            self.log_tail = (self.log_tail+text)[-3000:]
            with (self.job_directory / 'worker.log').open('a') as stream:
                stream.write(text)

    def process_error(self, error):
        if error == QProcess.FailedToStart:
            self.finished(-1, QProcess.CrashExit)

    def finished(self, exit_code, _exit_status):
        if self.process is None:
            return
        self.read_output()
        process, self.process = self.process, None
        process.deleteLater()
        execute_requested = self._execute_after_plan
        try:
            path = self.job_directory / 'result.json'
            result = json.loads(path.read_text()) if path.is_file() else {
                'ok': False, 'error': 'IK worker 未返回结果: '+self.log_tail}
            if self.sequence.state == 'stopped':
                self.message = '预检已取消，没有发送运动命令'
            elif not result.get('ok'):
                self.sequence.fail(result.get('error', f'Worker exit={exit_code}'))
                self.message = (('执行未启动：IK 预检失败：' if execute_requested else 'IK 预检失败：')
                                + self.sequence.error)
                self.execution_feedback.setText(self.message)
            else:
                self.sequence.accept_plan(result)
                hold_note = '；另一只手末端保持 base_link 位姿' if result.get('other_tcp_held') else ''
                self.message = (f"{result['stage']} IK/限位/连续性预检通过，约 {result['times_s'][-1]:.1f} 秒。"
                                f'本段路径确认已保留，可直接点击“执行本段”{hold_note}；未做自动碰撞检查。')
        except Exception as exc:
            self.sequence.fail(str(exc))
            self.message = ('执行未启动：' if execute_requested else '') + str(exc)
            self.execution_feedback.setText(self.message)
        self.execution_feedback.setText(self.message)
        if self.sequence.state in ('complete', 'stopped', 'failed'):
            self.set_reserved(False)
        if self._workflow_waiting and self.sequence.state == 'failed':
            self._workflow_waiting = False
        execute_after = self._execute_after_plan
        self._execute_after_plan = False
        if execute_after and self.sequence.state == 'planned':
            if self._workflow_auto_running and self._workflow_auto_paused:
                self.execution_feedback.setText('IK 预检通过；全部流程已暂停，等待继续。')
                self.message = self.execution_feedback.text()
            else:
                self.execution_feedback.setText('IK 预检通过，正在启动执行。')
                self._execute_ready_plan()
        elif self.sequence.state == 'planned':
            self.set_reserved(False)
        elif self._workflow_auto_running and self.sequence.state == 'failed':
            self._workflow_auto_paused = True
            self._workflow_waiting = False
        if self._workflow_auto_cancel_requested and self.process is None and self.runner is None:
            QTimer.singleShot(0, self._begin_auto_workflow_restore)
        self.refresh()

    def stop(self):
        if self._workflow_auto_running and not self._workflow_auto_cancel_requested:
            self.cancel_all_workflow()
            return
        self._execute_after_plan = False
        self._restoring = False
        self._restore_hands_goal = None
        self._wrench_guard_triggered = None
        self._workflow_waiting = False
        if self.sequence.state == 'stop_unconfirmed':
            self.message = '保持位置未获反馈确认，保留控制锁。请使用机器人停止装置并检查状态。'
            self.refresh()
            return
        self.sequence.stop(motion_active=self.runner is not None)
        if self.process:
            self.process.terminate()  # Read-only IK worker; it never sends commands.
        if self.runner:
            self.runner.stop()
        elif self.process is None:
            self.set_reserved(False)
            self.observation = None
        self.message = '已取消后续阶段；正在移动时保持实测位置并等待停止反馈，不会自动松手'
        self.refresh()

    def monitor(self):
        self.capture_startup_state()
        if self.runner:
            if not self.is_simulation and self.runner.state == 'running':
                side = self.runner.plan.get('arm_side', 'right')
                sides = (('left', 'right') if (side == 'both' or self.runner.plan.get('other_tcp_held'))
                         else (side,))
                violation = self.wrench_guard_violation(sides)
                if violation:
                    self._wrench_guard_triggered = violation
                    if self._workflow_auto_running:
                        self._workflow_auto_paused = True
                    self.runner.pause()
                    self.message = '六维力保护触发，正在暂停并保持位置：' + violation
            self.runner.tick()
            if self.runner.state in ('pausing', 'paused'):
                if self._wrench_guard_triggered:
                    self.message = ('六维力保护触发，已暂停并保持位置：' + self._wrench_guard_triggered
                                    if self.runner.state == 'paused' else
                                    '六维力保护触发，正在暂停并保持位置：' + self._wrench_guard_triggered)
                else:
                    self.message = self.runner.message
            if self.runner.state not in ('running', 'pausing', 'paused', 'stopping'):
                state, message = self.runner.state, self.runner.message
                was_auto_restore = self._workflow_auto_restoring
                if self._restoring:
                    if state == 'complete':
                        try:
                            message = self.finish_restore_hands()
                        except Exception as exc:
                            state, message = 'failed', str(exc)
                    self._restoring = False
                    self._workflow_auto_restoring = False
                    self._restore_hands_goal = None
                self.runner = None
                if not self.is_simulation:
                    self.node._stop_slow_motion_mode_timer()
                    # Stay in model mode; do not unexpectedly restore teleoperation.
                    self.node._slow_motion_saved_control_mode = None
                if state == 'complete':
                    self.sequence.motion_finished(True)
                elif state == 'stopped':
                    self.sequence.motion_finished(True, message)
                else:
                    self.sequence.fail(message)
                    if state == 'stop_unconfirmed':
                        self.sequence.state = state
                self.message = message
                if not self._workflow_auto_running:
                    self.scene_confirm.setChecked(False)
                if state == 'complete':
                    self.set_reserved(False)
                if self.sequence.state in ('complete', 'stopped', 'failed'):
                    self.set_reserved(False)
                    self.observation = None  # A physical movement consumes the observation.
                if was_auto_restore:
                    self._finish_auto_workflow_cancel(state == 'complete', message or state)
                elif self._workflow_auto_cancel_requested:
                    QTimer.singleShot(0, self._begin_auto_workflow_restore)
                elif self._workflow_waiting:
                    self._workflow_waiting = False
                    if state == 'complete':
                        self._advance_task_workflow()
                    else:
                        self.message = '流程步骤未完成：' + (message or state)
                        if self._workflow_auto_running:
                            self._workflow_auto_paused = True
        if self.process and self.sequence.state != 'stopped' and time.monotonic()-self.started > 180:
            self.stop()
            self.message = 'IK 预检超时，已取消'
            self.execution_feedback.setText(self.message)
        self.refresh()

    def refresh(self):
        state, stage = self.sequence.state, self.sequence.stage
        busy = self.process is not None or self.runner is not None
        auto_locked = self._workflow_auto_running
        self.task_combo.setEnabled(not busy and not self._workflow_waiting and not auto_locked)
        self.task_new_button.setEnabled(not busy and not self._workflow_waiting and not auto_locked)
        self.task_copy_button.setEnabled(not busy and not self._workflow_waiting and not auto_locked)
        self.task_manage_button.setEnabled(not busy and not self._workflow_waiting and not auto_locked)
        self.workflow_next_button.setEnabled(not busy and not self._workflow_waiting and not auto_locked)
        self.workflow_reset_button.setEnabled(not busy and not self._workflow_waiting and not auto_locked)
        self.task_save_button.setEnabled(not busy and not self.reserved and not auto_locked)
        self.workflow_all_button.setText('取消执行' if auto_locked else '全部执行')
        self.workflow_all_button.setEnabled(not self._workflow_auto_cancel_requested)
        self.workflow_auto_pause_button.setText('继续执行' if self._workflow_auto_paused else '暂停')
        pause_ready = (not self._workflow_auto_paused or self.runner is None
                       or self.runner.state == 'paused')
        self.workflow_auto_pause_button.setEnabled(
            auto_locked and not self._workflow_auto_cancel_requested and pause_ready)
        self.grasp_target_combo.setEnabled(not busy and not self.reserved and not auto_locked)
        for combo in self.task_role_combos.values():
            combo.setEnabled(not busy and not self._workflow_waiting and not auto_locked)
        guard_editable = self.runner is None or self.runner.state == 'paused'
        for widget in (self.wrench_guard_check, self.force_limit, self.torque_limit):
            widget.setEnabled(guard_editable and self.process is None and not auto_locked)
        self.restore_button.setEnabled(not busy and state != 'stop_unconfirmed')
        self.motion_stage.setEnabled(not busy and state != 'stop_unconfirmed' and not auto_locked)
        self.execution_mode.setEnabled(not busy and not self.reserved and not auto_locked)
        self.simulation_controls_button.setEnabled(self.is_simulation and not busy)
        runner_state = self.runner.state if self.runner is not None else ''
        self.pause_button.setEnabled(runner_state in ('running', 'paused'))
        self.pause_button.setText('继续执行' if runner_state == 'paused' else ('暂停中…' if runner_state == 'pausing' else '暂停'))
        self.plan_validity.setEnabled(not busy and not self.reserved)
        self.save_validity_button.setEnabled(not busy and not self.reserved)
        expired = False
        if state == 'planned' and self.sequence.plan:
            remaining = self.sequence.profile['max_plan_age_s'] - (time.monotonic() - self.sequence.plan['accepted_at'])
            expired = remaining < 0
            seconds = max(0, math.ceil(remaining))
            self.plan_expiry_label.setText(
                '规划已过期；点击执行本段会自动重新预检' if expired else
                f'规划剩余有效期：{seconds // 60:02d}:{seconds % 60:02d}（分:秒）')
        elif state == 'executing':
            self.plan_expiry_label.setText('本段已暂停，轨迹计时冻结' if runner_state in ('pausing', 'paused') else
                                          '本段执行中；有效期仅限制开始执行前的等待时间')
        else:
            self.plan_expiry_label.setText('尚无待执行计划；预检通过后开始倒计时')
        self.source_combo.setEnabled(not busy and not self.reserved and bool(self.method_observations))
        self.grasp_arm.setEnabled(not busy and not self.reserved and not auto_locked)
        self.waist_participation.setEnabled(not busy and not self.reserved and not auto_locked)
        self.suggest_shape_button.setEnabled(not busy and not self.reserved)
        self.apply_shape_button.setEnabled(not busy and not self.reserved and self.hand_shape_suggestion is not None)
        self.preview_button.setEnabled(not busy and not self.reserved and self.observation is not None)
        self.import_button.setEnabled(not busy and not self.reserved and self.observation is not None)
        self.manual_candidate_button.setEnabled(not busy and not self.reserved and self.observation is not None)
        for spin in self.contact_offsets:
            spin.setEnabled(not busy and not self.reserved)
        self.candidates.setEnabled(not busy and not self.reserved)
        self.profile_path.setEnabled(not busy and not self.reserved)
        self.browse_profile_button.setEnabled(not busy and not self.reserved)
        self.perception_button.setEnabled(not busy and not self.reserved)
        self.feedback_button.setEnabled(not busy and not self.reserved)
        self.calibration_button.setEnabled(not busy and not self.reserved)
        reason = ('请先到「分割」页完成一次有效的彩色图点击估计' if self.observation is None
                  else '正在执行或已锁定目标，请先停止当前任务')
        for button in (self.preview_button, self.manual_candidate_button, self.import_button):
            button.setToolTip(reason if not button.isEnabled() else '仅准备/预览，不会自动移动机器人')
        self.scene_confirm.setEnabled(not busy)
        self.hand_button.setEnabled(not busy and state != 'stop_unconfirmed')
        self.hand_button.setText('闭合手' if stage == 'close' else '张开手')
        self.confirm_button.setEnabled(not busy and state != 'stop_unconfirmed')
        self.confirm_button.setText('确认已抓住物体' if state == 'await_grasp' else '确认手已张开')
        self.plan_button.setEnabled(not busy and self.observation is not None and state != 'stop_unconfirmed')
        self.execute_button.setEnabled(not busy and self.observation is not None and state != 'stop_unconfirmed')
        if self.process is not None and self._execute_after_plan:
            self.execute_button.setText('自动预检中…')
        elif self.runner is not None:
            self.execute_button.setText('本段执行中…')
        else:
            self.execute_button.setText('执行本段（仿真）' if self.is_simulation else '执行本段（真机）')
        self.execute_button.setToolTip('执行所选阶段；如无有效计划，自动计算目标和预检，通过后执行。不自动开合手。')
        self.stop_button.setEnabled(self.sequence.preview is not None or busy)
        states = {'idle': '待开始', 'await_open': '等待确认张开', 'opened': '手已张开',
                  'planning': '预检中', 'planned': '等待执行确认', 'executing': '移动中',
                  'at_pregrasp': '已到预抓取位置', 'at_grasp': '已到抓取位置',
                  'await_grasp': '等待确认抓住', 'grasp_confirmed': '已确认夹持',
                  'complete': '已完成抬起', 'stopping': '正在停止', 'stopped': '已停止',
                  'failed': '失败', 'stop_unconfirmed': '停止未确认'}
        stages = {'open': '张开手', 'pregrasp': '预抓取位置', 'approach': '接近物体', 'close': '闭合手', 'lift': '抬起', 'restore': '恢复启动状态'}
        if runner_state in ('pausing', 'paused'):
            states['executing'] = '正在暂停' if runner_state == 'pausing' else '已暂停'
        display_stage = stage or (getattr(self.sequence, 'pending_stage', None) if state in ('planning', 'planned', 'executing') else None)
        startup = self._startup_snapshot or {}
        startup_state = ('双臂+双手已记录' if startup.get('hands') else
                         ('仅双臂已记录，双手缺失' if startup else '尚未记录'))
        top_step = self.task_flow.current_step or {}
        action = self._workflow_action()
        workflow_text = top_step.get('label', '已完成')
        if top_step.get('kind') == 'sequence' and action:
            workflow_text += ' → ' + action['label']
        lines = [('仿真执行（仅运动学播放，不控制真机）' if self.is_simulation else '真机执行'),
                 '启动恢复状态: '+startup_state,
                 ('腰部参与 IK 与执行' if self.waist_participation.isChecked() else '腰部固定'),
                 f"任务: {self.task_combo.currentText()}；当前流程: {workflow_text}",
                 f"状态: {states.get(state, state)}；当前/下一阶段: {stages.get(display_stage, '--')}",
                 '手动分步模式：未接入自动避障', self.message]
        if self.observation:
            lines.append('输入来源：'+self.observation.get('source_label', '彩色图点击估计'))
            lines.extend(self.observation.get('execution_blockers', []))
            lines.append('观测 ID: '+str(self.observation.get('observation_id', '缺失')))
        if self.sequence.preview:
            p = self.sequence.preview
            lines.append(f"手臂: {p['arm_side']} / 手型: {p['hand_model']} / TCP: {p['tcp_link']} / {p['base_frame']}")
            for name, T in p['goals'].items():
                lines.append(f'{name} XYZ(m): '+str(np.asarray(T)[:3, 3].round(4).tolist()))
            lines.append('TCP 旋转矩阵: '+np.array2string(np.asarray(p['goals']['approach'])[:3, :3], precision=3))
        if state == 'complete' and getattr(self.sequence, 'pending_stage', None) != 'restore':
            lines.append('已完成抬起到位；抓取成功由人工确认，不含力/触觉验证。可使用手动开合释放，重新抓取须重新估计。')
        text = '\n'.join(lines)
        if self.report.toPlainText() != text:
            self.report.setPlainText(text)
