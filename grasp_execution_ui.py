"""Qt grasp control panel: explicit stages, no execution on pose arrival."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget, QGridLayout)

from grasp_execution import Sequence, JointRunner, load_profile, make_preview, read_feedback, manual_candidate


class GraspExecutionPanel(QWidget):
    def __init__(self, node, busy, parent=None, preview_image=None, open_perception=None,
                 fill_extrinsic=None, ensure_feedback=None, open_calibration=None):
        super().__init__(parent)
        self.node, self.other_busy = node, busy
        self.preview_image_callback = preview_image
        self.open_perception = open_perception
        self.fill_extrinsic = fill_extrinsic
        self.ensure_feedback = ensure_feedback
        self.open_calibration = open_calibration
        self.sequence = Sequence()
        self.observation = None
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
        hint = QLabel('选择候选 → 预览 → 张开 → 预抓取 → 接近 → 闭合确认 → 抬起')
        hint.setWordWrap(True)
        layout.addWidget(hint)
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
        self.suggest_shape_button.clicked.connect(self.suggest_shape)
        shape_row.addWidget(self.suggest_shape_button)
        self.apply_shape_button = QPushButton('采用已标定手型')
        self.apply_shape_button.clicked.connect(self.apply_shape)
        shape_row.addWidget(self.apply_shape_button)
        layout.addLayout(shape_row)
        self.scene_confirm = QCheckBox('手动分步模式：当前无自动避障，已确认本段路径及手周围无障碍')
        layout.addWidget(self.scene_confirm)
        row = QGridLayout()
        self.preview_button = QPushButton('预览抓取位置')
        self.hand_button = QPushButton('张开手')
        self.confirm_button = QPushButton('确认手已张开')
        self.plan_button = QPushButton('预检下一段 IK')
        self.execute_button = QPushButton('执行本段')
        self.stop_button = QPushButton('停止 / 取消')
        for index, (button, callback) in enumerate(((self.preview_button, self.preview), (self.hand_button, self.hand),
                                 (self.confirm_button, self.confirm), (self.plan_button, self.plan),
                                 (self.execute_button, self.execute), (self.stop_button, self.stop))):
            button.clicked.connect(callback)
            button.setMinimumHeight(34)
            row.addWidget(button, index // 3, index % 3)
        self.stop_button.setStyleSheet('QPushButton { border: 1px solid #c65b55; font-weight: 600; }')
        layout.addLayout(row)
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
        self.candidates.currentIndexChanged.connect(self.invalidate_preview)
        for spin in self.contact_offsets:
            spin.valueChanged.connect(self.offset_changed)
        self.check_readiness()
        self.refresh()

    def suggest_shape(self):
        from object_hand_shape import suggest_hand_shape
        try:
            self.hand_shape_suggestion = suggest_hand_shape(self.observation)
            item = self.hand_shape_suggestion
            dims = ' × '.join(f'{x*1000:.0f}' for x in item['extents_m'])
            self.hand_shape_label.setText(f"建议：{item['name']}；参与手指：{'、'.join(item['fingers'])}\n"
                f"估计尺寸：{dims} mm；建议预张开约 {item['suggested_aperture_m']*1000:.0f} mm（需实测手型核对）\n"
                + item['note'] + '\n' + item['limitations'])
        except Exception as exc:
            self.hand_shape_suggestion = None
            self.hand_shape_label.setText('手型建议不可用：'+str(exc))
        self.refresh()

    def apply_shape(self):
        from object_hand_shape import calibrated_preset
        try:
            if self.reserved or self.process or not self.hand_shape_suggestion:
                raise ValueError('请先估计手型并结束当前动作')
            profile = load_profile(self.profile_path.text().strip())
            opened, closed = calibrated_preset(profile, self.hand_shape_suggestion)
            self.shape_override = (self.observation['observation_id'], self.profile_path.text().strip(), opened, closed)
            self.sequence = Sequence()
            self.message = '已选择该物体的标定手型；重新生成候选并预览后生效，没有发送开合命令。'
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def selected_profile(self, require_calibration=True):
        profile = load_profile(self.profile_path.text().strip(), require_calibration=require_calibration)
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
            profile = load_profile(path, require_calibration=False)
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
            lines.append('✓ 实测关节反馈新鲜')
        except Exception as exc:
            lines.append('反馈未就绪：'+str(exc))
            lines.append('可点“启动 FK 反馈”；它只订阅关节并发布 FK 状态，不发送手臂命令。')
        if not self.node.is_arm_enabled():
            lines.append('手臂未使能：执行前到「手动控制」启用手臂。')
        if not self.node.is_hand_enabled():
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
        self.message = '已收到新观测，已更新物体手型建议；请核对后预览。'
        self.refresh()

    @property
    def reserved(self):
        return bool(getattr(self.node, '_grasp_execution_reserved', False))

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
            result = make_preview(self.observation, self.candidates.currentIndex(), profile)
            self.sequence.reset(result, profile)
            if self.preview_image_callback is not None:
                pixmap = self.preview_image_callback(result)
                if pixmap.isNull():
                    self.preview_image.setText('使用手动控制里的融合6D结果；图像叠加请在来源页查看，下方为机器人坐标目标。')
                else:
                    self.preview_image.setPixmap(pixmap)
            extra = '' if profile.get('calibrated') else ' 当前 T_grasp_tcp 未标定，预览仅供核对，不能执行。'
            self.message = '预览完成。下一步检查并张开手；此时尚未验证机械臂轨迹。' + extra
        except Exception as exc:
            self.sequence.preview = None
            self.message = str(exc)
        self.refresh()

    def reserve(self):
        if self.other_busy():
            raise ValueError('已有手臂移动/文本队列/遥控，请先结束，不能同时控制')
        if not self.scene_confirm.isChecked():
            raise ValueError('本阶段尚未接入自动避障，请先确认本段路径周围无障碍')
        if not self.node.is_arm_enabled() or not self.node.is_hand_enabled():
            raise ValueError('请先手动启用手臂和手；抓取流程不自动使能')
        if not self.sequence.preview:
            raise ValueError('请先预览抓取位置')
        if not self.sequence.profile.get('calibrated'):
            raise ValueError('抓取配置尚未完成手型/TCP 标定，不能发送开合或手臂运动')
        self.node._grasp_execution_reserved = True

    def hand(self):
        try:
            self.reserve()
            stage = self.sequence.stage
            if stage not in ('open', 'close'):
                raise ValueError('当前不是开合阶段')
            self.feedback()
            values = self.sequence.preview['open_positions' if stage == 'open' else 'close_positions']
            self.node._grasp_hand_command_authorized = True
            try:
                self.node.apply_hand_joint_positions(self.sequence.profile['arm_side'], values)
            finally:
                self.node._grasp_hand_command_authorized = False
            self.sequence.hand_sent()
            self.message = '开合命令已发送；请观察实物，确认张开或确实抓住后再继续。'
        except Exception as exc:
            if self.sequence.state == 'planning' and self.process is None:
                self.sequence.fail(str(exc))
            self.message = str(exc)
        self.refresh()

    def confirm(self):
        try:
            self.sequence.confirm_hand()
            self.message = ('请规划预抓取路径' if self.sequence.state == 'opened'
                            else '已人工确认抓住；请预检抬起路径。没有力反馈，系统不能自动证明抓取成功。')
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def plan(self):
        try:
            self.reserve()
            stage = self.sequence.begin_plan()
            feedback = self.feedback()
            self.launch('plan', stage=stage, start_joints=feedback['q16'])
        except Exception as exc:
            if self.sequence.state == 'planning' and self.process is None:
                self.sequence.fail(str(exc))
                self.node._grasp_execution_reserved = False
            self.message = str(exc)
        self.refresh()

    def execute(self):
        try:
            self.reserve()
            plan = self.sequence.begin_execute(manual_path_confirmed=self.scene_confirm.isChecked())
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
            self.message = str(exc)
        self.refresh()

    def launch(self, mode, **extra):
        if self.process is not None:
            raise ValueError('已有规划/执行请求')
        self.mode = mode
        self.job_directory = self.root / '.cache' / 'grasp_execution' / uuid.uuid4().hex
        self.job_directory.mkdir(parents=True)
        request = dict(mode=mode, profile=self.sequence.profile, preview=self.sequence.preview, **extra)
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
        try:
            path = self.job_directory / 'result.json'
            result = json.loads(path.read_text()) if path.is_file() else {
                'ok': False, 'error': 'IK worker 未返回结果: '+self.log_tail}
            if self.sequence.state == 'stopped':
                self.message = '预检已取消，没有发送运动命令'
            elif not result.get('ok'):
                self.sequence.fail(result.get('error', f'Worker exit={exit_code}'))
                self.message = self.sequence.error
            else:
                self.sequence.accept_plan(result)
                self.scene_confirm.setChecked(False)
                self.message = (f"{result['stage']} IK/限位/连续性预检通过，约 {result['times_s'][-1]:.1f} 秒。"
                                '未做碰撞检查；核对路径后点击执行。')
        except Exception as exc:
            self.sequence.fail(str(exc))
            self.message = str(exc)
        if self.sequence.state in ('complete', 'stopped', 'failed'):
            self.node._grasp_execution_reserved = False
        self.refresh()

    def stop(self):
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
            self.node._grasp_execution_reserved = False
            self.observation = None
        self.message = '已取消后续阶段；正在移动时保持实测位置并等待停止反馈，不会自动松手'
        self.refresh()

    def monitor(self):
        if self.runner:
            self.runner.tick()
            if self.runner.state not in ('running', 'stopping'):
                state, message = self.runner.state, self.runner.message
                self.runner = None
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
                self.scene_confirm.setChecked(False)
                if self.sequence.state in ('complete', 'stopped', 'failed'):
                    self.node._grasp_execution_reserved = False
                    self.observation = None  # A physical movement consumes the observation.
        if self.process and self.sequence.state != 'stopped' and time.monotonic()-self.started > 180:
            self.stop()
            self.message = 'IK 预检超时，已取消'
        self.refresh()

    def refresh(self):
        state, stage = self.sequence.state, self.sequence.stage
        busy = self.process is not None or self.runner is not None
        self.suggest_shape_button.setEnabled(not busy and not self.reserved and self.observation is not None)
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
        self.hand_button.setEnabled(not busy and self.sequence.preview is not None and stage in ('open', 'close'))
        self.hand_button.setText('闭合手' if stage == 'close' else '张开手')
        self.confirm_button.setEnabled(not busy and state in ('await_open', 'await_grasp'))
        self.confirm_button.setText('确认已抓住物体' if state == 'await_grasp' else '确认手已张开')
        self.plan_button.setEnabled(not busy and stage in ('pregrasp', 'approach', 'lift'))
        self.execute_button.setEnabled(not busy and state == 'planned')
        self.stop_button.setEnabled(self.sequence.preview is not None or busy)
        states = {'idle': '待开始', 'await_open': '等待确认张开', 'opened': '手已张开',
                  'planning': '预检中', 'planned': '等待执行确认', 'executing': '移动中',
                  'at_pregrasp': '已到预抓取位置', 'at_grasp': '已到抓取位置',
                  'await_grasp': '等待确认抓住', 'grasp_confirmed': '已确认夹持',
                  'complete': '已完成抬起', 'stopping': '正在停止', 'stopped': '已停止',
                  'failed': '失败', 'stop_unconfirmed': '停止未确认'}
        stages = {'open': '张开手', 'pregrasp': '预抓取位置', 'approach': '接近物体', 'close': '闭合手', 'lift': '抬起'}
        display_stage = stage or (getattr(self.sequence, 'pending_stage', None) if state in ('planning', 'planned', 'executing') else None)
        lines = [f"状态: {states.get(state, state)}；当前/下一阶段: {stages.get(display_stage, '--')}",
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
        if state == 'complete':
            lines.append('已完成抬起到位；抓取成功由人工确认，不含力/触觉验证。可使用手动开合释放，重新抓取须重新估计。')
        text = '\n'.join(lines)
        if self.report.toPlainText() != text:
            self.report.setPlainText(text)
