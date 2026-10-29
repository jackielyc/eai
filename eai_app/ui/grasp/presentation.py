from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.grasp.dependencies')

class PresentationMixin:
    def refresh(self):
        if hasattr(self,'execution_feedback'):self.execution_feedback.setText(self.message)
        state, stage = self.sequence.state, self.sequence.stage
        busy = self.process is not None or self.runner is not None or self.hand_runner is not None or self._perception_pending or self._program_active
        auto_locked = self._workflow_auto_running
        for widget in (self.scene_edit_button,self.assembly_edit_button,self.rank_candidates_button,self.runtime_verify_button,self.motion_limits_button,self.attach_button,self.empty_hand_button):
            widget.setEnabled(not busy and not auto_locked and not self._workflow_waiting and not self.reserved)
        self.collision_check.setEnabled(self.is_simulation and not busy and not self.reserved and not auto_locked)
        self.reestimate_check.setEnabled(not busy and not self.reserved and not auto_locked)
        workflow_status = self.workflow_status()
        self.workflow_status_label.setText(workflow_status)
        self.task_flow.set_completed_steps(self._workflow_completed_steps)
        flow_state = ('failed' if self._workflow_error or state == 'stop_unconfirmed' else
                      'paused' if self._workflow_auto_paused else
                      'confirm' if self._workflow_confirmation else 'active')
        self.task_flow.set_runtime_status(workflow_status, flow_state)
        self.workflow_confirm_button.setEnabled(
            bool(self._workflow_confirmation) and not busy and not self._workflow_auto_paused
            and not self._workflow_auto_cancel_requested and state != 'stop_unconfirmed')
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
        if self.hand_runner and self._workflow_auto_paused:
            pause_ready = self.hand_runner.state == 'paused'
        self.workflow_auto_pause_button.setEnabled(
            auto_locked and not self._workflow_auto_cancel_requested and pause_ready
            and state != 'stop_unconfirmed')
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
        pause_state = self.hand_runner.state if self.hand_runner is not None else runner_state
        self.pause_button.setEnabled(pause_state in ('running', 'paused'))
        self.pause_button.setText('继续执行' if pause_state == 'paused' else ('暂停中…' if pause_state == 'pausing' else '暂停'))
        self.plan_validity.setEnabled(not busy and not self.reserved)
        self.save_validity_button.setEnabled(not busy and not self.reserved)
        expired = False
        if state == 'planned' and self.sequence.plan:
            remaining = self.sequence.profile['max_plan_age_s'] - (_deps.time.monotonic() - self.sequence.plan['accepted_at'])
            expired = remaining < 0
            seconds = max(0, _deps.math.ceil(remaining))
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
        self.hand_button.setEnabled(not busy and not auto_locked and not self._workflow_waiting and state != 'stop_unconfirmed')
        self.hand_button.setText('闭合手' if stage == 'close' else '张开手')
        self.confirm_button.setEnabled(not busy and not auto_locked and not self._workflow_waiting and state != 'stop_unconfirmed')
        self.confirm_button.setText('确认已抓住物体' if state == 'await_grasp' else '确认手已张开')
        self.plan_button.setEnabled(not busy and not auto_locked and not self._workflow_waiting and self.observation is not None and state != 'stop_unconfirmed')
        self.execute_button.setEnabled(not busy and not auto_locked and not self._workflow_waiting and self.observation is not None and state != 'stop_unconfirmed')
        if self.process is not None and self._execute_after_plan:
            self.execute_button.setText('自动预检中…')
        elif self.runner is not None:
            self.execute_button.setText('本段执行中…')
        else:
            self.execute_button.setText('执行本段（仿真）' if self.is_simulation else '执行本段（真机）')
        self.execute_button.setToolTip('执行所选阶段；如无有效计划，自动计算目标和预检，通过后执行。不自动开合手。')
        self.stop_button.setEnabled(self.sequence.preview is not None or busy or self._workflow_waiting or auto_locked)
        if self._workflow_waiting or auto_locked:
            for widget in (self.restore_button, self.execution_mode, self.source_combo,
                           self.simulation_controls_button, self.preview_button,
                           self.manual_candidate_button, self.import_button,
                           self.suggest_shape_button, self.apply_shape_button,
                           self.profile_path, self.browse_profile_button, self.scene_confirm,
                           self.candidates, *self.contact_offsets):
                widget.setEnabled(False)
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
                 '流程状态: ' + workflow_status,
                 f"机械臂阶段状态: {states.get(state, state)}；当前/下一阶段: {stages.get(display_stage, '--')}",
                 ('静态场景采样碰撞复检；不含动态避障' if self.use_execution_service and self.collision_check.isChecked()
                  else '运动学模式：未进行碰撞复检'), self.message]
        if self.observation:
            lines.append('输入来源：'+self.observation.get('source_label', '彩色图点击估计'))
            lines.extend(self.observation.get('execution_blockers', []))
            lines.append('观测 ID: '+str(self.observation.get('observation_id', '缺失')))
        if self.run_log.path:
            lines.append('执行记录: '+str(self.run_log.path))
        if self._run_log_error:
            lines.append(self._run_log_error)
        if self.sequence.preview:
            p = self.sequence.preview
            lines.append(f"手臂: {p['arm_side']} / 手型: {p['hand_model']} / TCP: {p['tcp_link']} / {p['base_frame']}")
            for name, T in p['goals'].items():
                lines.append(f'{name} XYZ(m): '+str(_deps.np.asarray(T)[:3, 3].round(4).tolist()))
            lines.append('TCP 旋转矩阵: '+_deps.np.array2string(_deps.np.asarray(p['goals']['approach'])[:3, :3], precision=3))
        if state == 'complete' and getattr(self.sequence, 'pending_stage', None) != 'restore':
            lines.append('已完成抬起到位；抓取成功由人工确认，不含力/触觉验证。可使用手动开合释放，重新抓取须重新估计。')
        text = '\n'.join(lines)
        if self.report.toPlainText() != text:
            self.report.setPlainText(text)

