from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.grasp.dependencies')

class ConfigurationControllerMixin:
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
            violation, force_norm, torque_norm = _deps.evaluate_wrench(
                sample, self.force_limit.value(), self.torque_limit.value())
            label = '左腕' if side == 'left' else '右腕'
            if force_norm is not None:
                reports.append(f'{label} |F|={force_norm:.2f}N |M|={torque_norm:.3f}N·m')
            if violation:
                self.wrench_guard_status.setText(label + '：' + violation)
                return label + '：' + violation
        self.wrench_guard_status.setText('；'.join(reports) if reports else '等待六维力反馈')
        return None

    def load_plan_validity(self, *_):
        if not self.reserved:
            self.simulation = None
        try:
            profile = _deps.load_profile(self.profile_path.text().strip(), require_calibration=False)
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
        self.sequence = _deps.Sequence()
        self.scene_confirm.setChecked(False)
        side = '左臂' if self.grasp_arm.currentData() == 'left' else '右臂'
        self.message = f'已选择{side}抓取；依赖旧手腕朝向的候选和计划已清除，请重新生成或预览。'
        self.check_readiness()
        self.refresh()

    @property
    def is_simulation(self):
        return self.execution_mode.currentData() == 'simulation'

    def change_execution_mode(self, *_):
        if (self.runner or self.process or self.hand_runner or self._program_active) and not self._program_backend_authorized:
            self.execution_mode.blockSignals(True)
            self.execution_mode.setCurrentIndex(self.execution_mode.findData(self._last_execution_mode))
            self.execution_mode.blockSignals(False)
            self.message='当前控制仍被占用，不能切换仿真/真机模式。';self.refresh();return
        self._last_execution_mode=str(self.execution_mode.currentData())
        self.simulation = None
        self._sim_reserved = False
        self.sequence = _deps.Sequence()
        self.shape_override = None
        if self.simulation_display:
            self.simulation_display(None, {})
        self.message = '执行方式已切换，请重新预览；仿真是运动学播放，不模拟物体接触。'
        self.refresh()

    def simulation_backend(self):
        if self.simulation is None:
            if self.simulation_display is None:
                raise ValueError('未连接机器人页面仿真显示')
            from eai_app.adapters.simulation.kinematic_playback import GraspSimulation
            profile = _deps.load_profile(self.profile_path.text().strip(), require_calibration=False)
            seed = (self._startup_snapshot or {}).get('q16', self._startup_seed)
            self.simulation = GraspSimulation(profile, seed, self.simulation_display)
            self.simulation.hands = _deps.copy.deepcopy((self._startup_snapshot or {}).get('hands', {'left':[0.]*6, 'right':[0.]*6}))
            self._simulation_startup = dict(self.simulation.feedback(), hands=_deps.copy.deepcopy(self.simulation.hands))
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
            path = _deps.Path(self.profile_path.text().strip())
            data = _deps.json.loads(path.read_text())
            data['max_plan_age_s'] = self.plan_validity.value()
            path.write_text(_deps.json.dumps(data, ensure_ascii=False, indent=2) + '\n')
            self.sequence = _deps.Sequence()
            self.message = f'规划有效期已保存为 {self.plan_validity.value():g} 秒；请重新预览。'
        except Exception as exc:
            self.message = '有效期未保存：' + str(exc)
        self.refresh()

    def selected_profile(self, require_calibration=True):
        profile = _deps.load_profile(self.profile_path.text().strip(), require_calibration=require_calibration and not self.is_simulation)
        side = str(self.grasp_arm.currentData() or profile['arm_side'])
        profile['arm_side'] = side
        profile['tcp_link'] = 'Link7_l' if side == 'left' else 'Link7_r'
        override = getattr(self, 'shape_override', None)
        if override and self.observation and override[:2] == (self.observation['observation_id'], self.profile_path.text().strip()):
            profile['open_positions'], profile['close_positions'] = override[2:]
        return profile

    def browse_profile(self):
        path, _ = _deps.QFileDialog.getOpenFileName(self, '选择已标定的抓取配置', str(self.root / 'config'), 'JSON (*.json)')
        if path:
            self.profile_path.setText(path)
            self.check_readiness()

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

    def check_readiness(self):
        lines = []
        profile = None
        try:
            path = _deps.Path(self.profile_path.text().strip())
            if not path.is_file():
                raise ValueError(f'抓取配置不存在：{path}。请先填写 config/grasp_execution.example.json 中的实际标定数据并另存；模板不能直接用于执行。')
            profile = self.selected_profile(require_calibration=False)
            lines.append(f"✓ 当前选择：{'左臂 / Link7_l' if profile['arm_side'] == 'left' else '右臂 / Link7_r'}")
            if profile.get('calibrated'):
                lines.append('✓ 抓取配置已通过格式和标定声明检查')
            else:
                lines.append('抓取配置格式可用，但尚未完成手型/TCP 标定：可以生成候选并预览，不能张开手或执行运动。')
            for key, label in (('planner_python', 'IK Python'), ('urdf_path', '机器人 URDF')):
                if not _deps.Path(profile[key]).is_file():
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

    def feedback(self):
        if self.is_simulation:
            return self.simulation_backend().feedback()
        return _deps.read_feedback(_deps.os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))

