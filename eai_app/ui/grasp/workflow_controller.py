from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.grasp.dependencies')

class WorkflowControllerMixin:
    def change_task_workflow(self, *_):
        task_key = str(self.task_combo.currentData() or 'phone_box_assembly')
        saved = None
        definition_changed=False
        if self.load_task_data:
            try:
                saved = self.load_task_data(task_key)
                if saved and saved.get('task_definition'):
                    cached_definition = _deps.validate_task(saved['task_definition'])
                    current_revision = int(_deps.TASKS.get(task_key, {}).get('definition_revision', 1))
                    if cached_definition.get('definition_revision', 1) >= current_revision:
                        _deps.TASKS[task_key] = cached_definition
                    else:definition_changed=True
            except Exception as exc:
                self.task_data_status.setText('任务数据加载失败：' + str(exc))
        task = _deps.TASKS[task_key]
        self._render_task_controls(task)
        self.reset_task_workflow()
        if saved:
            self.set_target_observations(saved.get('target_observations') or {})
            state = saved.get('panel_state') or {}
            self._workflow_step_settings = _deps.reconcile_settings(state.get('step_settings') or {},task,
                                                              allow_legacy=not definition_changed)
            if definition_changed:
                state=dict(state,workflow_step=0,workflow_substep=0,workflow_finished=False,completed_steps=[])
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
            self._workflow_completed_steps = {
                int(i) for i in state.get('completed_steps', [])
                if isinstance(i, int) and 0 <= i < len(task['steps'])}
            self.task_flow.set_current(self._workflow_step)
            if self._workflow_finished:
                self.task_flow.set_current(len(task['steps']))
            self._apply_workflow_step_settings(self._workflow_step, self._workflow_substep)
            saved_at = str(saved.get('saved_at_text') or '')
            self.task_data_status.setText('任务数据：已加载' + (f'（{saved_at}）' if saved_at else ''))
            if definition_changed:self.task_data_status.setText(self.task_data_status.text()+'；流程已升级，参数按动作匹配，执行进度重置。')
        elif self.load_task_data:
            self.set_target_observations({})
            self._workflow_step_settings = {}
            self.task_data_status.setText('任务数据：该任务尚无保存结果')
        if hasattr(self,'phone_program_panel'):self.phone_program_panel.refresh_visibility()

    def _render_task_controls(self, task):
        self.task_flow.set_task(task)
        while self.task_role_row.count():
            item = self.task_role_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.task_role_combos = {}
        for role, label in task['roles']:
            self.task_role_row.addWidget(_deps.QLabel(label+'目标'))
            combo = _deps.QComboBox()
            self.task_role_combos[role] = combo
            self.task_role_row.addWidget(combo)
        self.task_role_row.addStretch(1)
        self._refresh_task_role_targets()

    def refresh_task_selector(self, selected_key=''):
        selected_key = selected_key or str(self.task_combo.currentData() or '')
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        for key, task in _deps.TASKS.items():
            self.task_combo.addItem(task['name'], key)
        index = self.task_combo.findData(selected_key)
        self.task_combo.setCurrentIndex(index if index >= 0 else 0)
        self.task_combo.blockSignals(False)
        self.change_task_workflow()

    def _create_task_from_dialog(self, source=None):
        dialog = _deps.TaskEditorDialog(source, self)
        if dialog.exec_() != _deps.QDialog.Accepted:
            return
        try:
            definition = dialog.definition()
            key = _deps.new_task_key()
            _deps.TASKS[key] = definition
            _deps.save_tasks(_deps.TASKS)
            self.refresh_task_selector(key)
            self.task_data_status.setText('任务定义已创建；运行数据尚未保存')
        except Exception as exc:
            self.task_data_status.setText('任务定义创建失败：' + str(exc))
            _deps.QMessageBox.warning(self, '任务未创建', str(exc))

    def new_task(self):
        self._create_task_from_dialog(None)

    def copy_task(self):
        source = _deps.TASKS.get(str(self.task_combo.currentData() or ''))
        if not source:
            return
        name, accepted = _deps.QInputDialog.getText(
            self, '复制任务', '副本名称', text=source['name']+' 副本')
        if not accepted:
            return
        try:
            name = name.strip()
            if not name:
                raise ValueError('副本名称不能为空')
            definition = _deps.validate_task(dict(_deps.copy.deepcopy(source), name=name, builtin=False))
            key = _deps.new_task_key()
            _deps.TASKS[key] = definition
            _deps.save_tasks(_deps.TASKS)
            self.refresh_task_selector(key)
            self.task_data_status.setText('任务定义复制成功；运行数据尚未保存')
            self.message = f'已复制任务“{source["name"]}”为“{name}”'
            self.refresh()
        except Exception as exc:
            self.task_data_status.setText('任务复制失败：' + str(exc))
            _deps.QMessageBox.warning(self, '任务复制失败', str(exc))

    def manage_tasks(self):
        action, ok = _deps.QInputDialog.getItem(
            self, '管理任务', '操作', ('重命名当前任务', '删除当前任务', '导出当前任务', '导入任务'), 0, False)
        if not ok:
            return
        key = str(self.task_combo.currentData() or '')
        task = _deps.TASKS.get(key)
        try:
            if action == '重命名当前任务':
                name, accepted = _deps.QInputDialog.getText(self, '重命名任务', '新名称', text=task['name'])
                if accepted and name.strip():
                    _deps.TASKS[key] = _deps.validate_task(dict(task, name=name.strip()))
                    _deps.save_tasks(_deps.TASKS)
                    self.refresh_task_selector(key)
            elif action == '删除当前任务':
                if task.get('builtin'):
                    raise ValueError('内置任务不能删除，可以复制后修改')
                if _deps.QMessageBox.question(self, '删除任务', f"删除任务定义“{task['name']}”？\n已保存运行数据不会删除。") == _deps.QMessageBox.Yes:
                    _deps.TASKS.pop(key)
                    _deps.save_tasks(_deps.TASKS)
                    self.refresh_task_selector()
            elif action == '导出当前任务':
                path, _ = _deps.QFileDialog.getSaveFileName(self, '导出任务', task['name']+'.json', 'JSON (*.json)')
                if path:
                    _deps.Path(path).write_text(_deps.json.dumps({'version':1,'task':task}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            else:
                path, _ = _deps.QFileDialog.getOpenFileName(self, '导入任务', '', 'JSON (*.json)')
                if path:
                    data = _deps.json.loads(_deps.Path(path).read_text(encoding='utf-8'))
                    imported = _deps.validate_task(data.get('task') or data)
                    imported['builtin'] = False
                    new_key = _deps.new_task_key()
                    _deps.TASKS[new_key] = imported
                    _deps.save_tasks(_deps.TASKS)
                    self.refresh_task_selector(new_key)
        except Exception as exc:
            _deps.QMessageBox.warning(self, '任务管理失败', str(exc))

    def current_task_state(self):
        self._capture_workflow_step_settings(self._workflow_step, self._workflow_substep)
        return {
            'last_run_id': self.run_log.run_id,
            'world_snapshot': {'attached':_deps.copy.deepcopy(self.world.attached),'placed':_deps.copy.deepcopy(self.world.placed)},
            'roles': {role: str(combo.currentData() or '') for role, combo in self.task_role_combos.items()},
            'selected_target': str(self.grasp_target_combo.currentData() or ''),
            'workflow_step': self._workflow_step,
            'workflow_substep': self._workflow_substep,
            'workflow_finished': self._workflow_finished,
            'completed_steps': sorted(self._workflow_completed_steps),
            'arm_side': str(self.grasp_arm.currentData() or ''),
            'allow_waist': self.waist_participation.isChecked(),
            'motion_stage': str(self.motion_stage.currentData() or ''),
            'pose_method': str(self.source_combo.currentData() or ''),
            'step_settings': _deps.copy.deepcopy(self._workflow_step_settings),
        }

    def _workflow_action(self, index=None, substep=None):
        task = _deps.TASKS.get(str(self.task_combo.currentData() or ''))
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
            'step': _deps.copy.deepcopy(step),
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
        if self.runner or self.process or self.hand_runner or self._workflow_auto_running:
            return
        self._workflow_confirmation = None
        self._workflow_error = ''
        self.run_log.start()
        self._workflow_step = 0
        self._workflow_substep = 0
        self._workflow_finished = False
        self._workflow_completed_steps.clear()
        self._workflow_waiting = False
        self.task_flow.set_current(0)
        self.message = f"{self.task_combo.currentText()}：请选择任务目标，然后执行流程下一步。"
        self.refresh()

    def select_workflow_step(self, index):
        if (hasattr(self,'phone_program_panel') and self.phone_program_panel.is_current_task()
                and not self.assembly_advanced_check.isChecked()):
            self.phone_program_panel.inspect_stage(index);return
        if self.runner or self.process or self._workflow_waiting or self._workflow_auto_running:
            self.message = '当前正在规划或执行，不能跳转流程步骤。'
            self.refresh()
            return
        task = _deps.TASKS[str(self.task_combo.currentData())]
        index = max(0, min(int(index), len(task['steps'])-1))
        self._workflow_step = index
        self._workflow_confirmation = None
        self._workflow_error = ''
        self._workflow_substep = 0
        self._workflow_finished = False
        self.sequence = _deps.Sequence()
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

    def record_workflow_event(self, event, **details):
        try:
            plan = self.sequence.plan or {}
            context = {
                'task_key': str(self.task_combo.currentData()),
                'step': self._workflow_step, 'substep': self._workflow_substep,
                'action': self._workflow_action(),
                'mode': str(self.execution_mode.currentData()),
                'arm_side': str(self.grasp_arm.currentData()),
                'observation_id': (self.observation or {}).get('observation_id'),
                'preview_id': (self.sequence.preview or {}).get('id'),
                'urdf_sha256': plan.get('urdf_sha256'),
                'message': self.message, **details,
            }
            self.run_log.append(event, context)
        except Exception as exc:
            self._run_log_error = '执行记录写入失败：' + str(exc)

    def _advance_task_workflow(self):
        self.record_workflow_event('action_completed')
        self._workflow_confirmation = None
        self._workflow_error = ''
        task = _deps.TASKS[str(self.task_combo.currentData())]
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
            self._workflow_completed_steps.add(self._workflow_step)
            self._workflow_step += 1
            self._workflow_substep = 0
            self.task_flow.set_current(self._workflow_step)
            self._apply_workflow_step_settings(self._workflow_step, 0)
            step = task['steps'][self._workflow_step]
            action = self._workflow_action()
            self.message = f"流程进入：{step['label']}" + (f" → {action['label']}" if step['kind']=='sequence' else '')
        else:
            self._workflow_completed_steps.add(self._workflow_step)
            self._workflow_finished = True
            self.task_flow.set_current(len(task['steps']))
            self.message = f"{task['name']}：" + ('全部流程执行完成。' if self._workflow_auto_running else '流程已完成。')
            self.record_workflow_event('workflow_completed')
            self._workflow_auto_running = False
            self._workflow_auto_paused = False
            self._workflow_auto_initial_snapshot = None
        if (self._workflow_auto_running and not self._workflow_auto_cancel_requested
                and not self._workflow_finished):
            self._schedule_auto_workflow(1000 if completed_kind == 'hand' else 200)

    def _workflow_target(self, role):
        combo = self.task_role_combos.get(role)
        target_id = str(combo.currentData() or '') if combo else ''
        if not target_id or target_id not in self.target_observations:
            raise ValueError(f"请先选择{dict(_deps.TASKS[str(self.task_combo.currentData())]['roles'])[role]}目标")
        for other_role, other_combo in self.task_role_combos.items():
            other_id = str(other_combo.currentData() or '')
            if other_role != role and other_id and target_id == other_id:
                labels = dict(_deps.TASKS[str(self.task_combo.currentData())]['roles'])
                raise ValueError(f"{labels[role]}目标和{labels[other_role]}目标不能相同")
        self.select_target_observation(target_id)
        return target_id

    def _workflow_hand(self, opened):
        self._hand_purpose='workflow'
        self._startup_recording = False
        profile = self.selected_profile()
        self.reserve(require_preview=False)
        violation = self.wrench_guard_violation((profile['arm_side'],))
        if violation:
            raise ValueError('六维力保护阻止手部动作：'+violation)
        values = profile['open_positions' if opened else 'close_positions']
        side = profile['arm_side']
        if self.use_execution_service:
            self.hand_runner=self.new_process_runner('hand',side=side,target=values)
            self._workflow_waiting=True
            self._pending_hand_open=opened
            self._pending_hand_side=side
            self._pending_hand_role=(self._workflow_action() or {}).get('role')
            self._pending_hand_observation=_deps.copy.deepcopy(self.observation)
            self._pending_hand_target=str(self.grasp_target_combo.currentData() or '')
            self.message='独立服务执行手部动作，等待实际反馈'
            return
        simulation = self.is_simulation
        def publish(positions):
            if simulation:
                self.simulation_backend().hand(side, positions)
                return
            self.node._grasp_hand_command_authorized = True
            try:
                self.node.apply_hand_joint_positions(side, positions)
            finally:
                self.node._grasp_hand_command_authorized = False
        def feedback():
            if simulation:
                sim = self.simulation_backend()
                return {'stamp': _deps.time.time(), 'positions': sim.hands.get(side, [0.]*6)}
            return self.node.get_grasp_hand_feedback(side)
        runner = _deps.HandRunner(side, values, publish, feedback,
                            lambda: simulation or self.node.is_hand_enabled())
        self.hand_runner = runner
        try:
            runner.start()
        except Exception:
            if runner.state == 'ready':
                self.hand_runner = None
            else:
                self.sequence.state = 'stop_unconfirmed'
            raise
        self._workflow_waiting = True
        self.message = runner.message
        self.record_workflow_event('hand_command_sent', target=values)

    def confirm_workflow_action(self):
        token = (str(self.task_combo.currentData()), self._workflow_step, self._workflow_substep)
        if (self._workflow_confirmation != token or self.runner or self.process or self.hand_runner
                or self._workflow_auto_paused or self._workflow_auto_cancel_requested):
            return
        if self.use_execution_service and self.pending_attachment:
            try:
                item=self.pending_attachment
                self.world.attach_confirmed(item['side'],item['role'],item['id'],item['observation'],self.feedback())
                self._unverified_grips.discard(item['side'])
                self.pending_attachment=None
                self.record_workflow_event('object_attached',attached=self.world.attached)
            except Exception as exc:
                self.message='无法确认附着物：'+str(exc);self.refresh();return
        self._workflow_waiting = False
        self.record_workflow_event('operator_confirmed')
        if getattr(self,'_attachment_only_confirm',False):
            self._attachment_only_confirm=False;self._workflow_confirmation=None
            if self._workflow_auto_running:self._schedule_auto_workflow(0)
            self.message='已确认夹持；可继续当前动作。';self.refresh();return
        self._advance_task_workflow()
        self.refresh()

    def execute_workflow_next(self):
        if self.runner or self.process or self._workflow_waiting or self.sequence.state == 'stop_unconfirmed':
            return
        task = _deps.TASKS[str(self.task_combo.currentData())]
        step = self._workflow_action()
        if step is None or self._workflow_finished:
            self.message = f"{task['name']}流程已完成。"
            self.refresh()
            return
        try:
            self._workflow_error = ''
            self.record_workflow_event('action_started')
            self._capture_workflow_step_settings(self._workflow_step, self._workflow_substep)
            kind = step['kind']
            if kind == 'target':
                self._workflow_target(step['role'])
                self._advance_task_workflow()
            elif kind == 'motion':
                if self.use_execution_service and self.pending_attachment:
                    self._attachment_only_confirm=True
                    self._workflow_confirmation=(str(self.task_combo.currentData()),self._workflow_step,self._workflow_substep)
                    self._workflow_waiting=True;self.message='先确认夹持，建立物体与手的相对关系后再移动。'
                    self.refresh();return
                self._workflow_target(step['role'])
                self.motion_stage.setCurrentIndex(self.motion_stage.findData(step['stage']))
                self._workflow_waiting = True
                self.execute()
                if self.runner is None and self.process is None and not self._perception_pending:
                    self._workflow_waiting = False
                    self._workflow_error = self.message
                    self._workflow_auto_paused = self._workflow_auto_running
            elif kind == 'hand':
                self._workflow_target(step['role'])
                self._workflow_hand(step['open'])
            elif kind == 'confirm':
                self._workflow_confirmation = (str(self.task_combo.currentData()),
                                               self._workflow_step, self._workflow_substep)
                self._workflow_waiting = True
                self.message = '等待人工确认：' + step['label']
                self.record_workflow_event('confirmation_requested')
            elif kind == 'complete':
                self._advance_task_workflow()
        except Exception as exc:
            self._workflow_waiting = False
            if self.sequence.state != 'stop_unconfirmed':
                self.set_reserved(False)
            self.message = str(exc)
            self._workflow_error = self.message
            self.record_workflow_event('action_failed')
            if self._workflow_auto_running:
                self._workflow_auto_paused = True
        self.refresh()

    def _capture_auto_workflow_start(self):
        if self.is_simulation:
            simulation = self.simulation_backend()
            return dict(simulation.feedback(), hands=_deps.copy.deepcopy(simulation.hands))
        measured = _deps.copy.deepcopy(self.feedback())
        try:
            measured['hands'] = self.read_startup_hands()
        except Exception:
            measured['hands'] = {}
        return measured

    def toggle_all_workflow(self):
        if self.sequence.state == 'stop_unconfirmed':
            self.message = '停止尚未确认，请检查机器人；不能开始或自动恢复流程。'
            self.refresh()
            return
        if self._workflow_auto_running:
            self.cancel_all_workflow()
            return
        if self.runner or self.process or self.reserved:
            self.message = '请先停止当前动作，再执行全部流程。'
            self.refresh()
            return
        try:
            for role, label in _deps.TASKS[str(self.task_combo.currentData())]['roles']:
                combo = self.task_role_combos.get(role)
                if combo is None or not combo.currentData():
                    raise ValueError(f'请先选择{label}目标')
            self._workflow_step = 0
            self._workflow_substep = 0
            self._workflow_finished = False
            self._workflow_completed_steps.clear()
            self._workflow_waiting = False
            self.task_flow.set_current(0)
            self._apply_workflow_step_settings(0, 0)
            if not self.is_simulation and not self.scene_confirm.isChecked():
                raise ValueError('真机全部执行前，请确认完整流程路径及手周围无障碍')
            self._workflow_auto_initial_snapshot = self._capture_auto_workflow_start()
            self.run_log.start()
            self.record_workflow_event('workflow_started',
                task_definition=_deps.copy.deepcopy(_deps.TASKS[str(self.task_combo.currentData())]),
                initial_state=self._workflow_auto_initial_snapshot)
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
        task = _deps.TASKS[str(self.task_combo.currentData())]
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
            if self.hand_runner is not None:
                try:
                    violation = self.wrench_guard_violation((self.hand_runner.side,))
                    if violation:
                        raise ValueError('六维力仍不满足继续条件：'+violation)
                    self.hand_runner.resume()
                except Exception as exc:
                    self.message = str(exc)
                    self.refresh()
                    return
            if self.runner is not None and self.runner.state == 'paused':
                self.pause_or_resume()
                if self.runner.state not in ('running','resuming'):
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
            if self.hand_runner is not None:
                self.hand_runner.pause()
            if self.runner is not None and self.runner.state == 'running':
                self.pause_or_resume()
            self.message = '全部流程已暂停；当前轨迹保持位置。'
        self.record_workflow_event('workflow_paused' if self._workflow_auto_paused else 'workflow_resumed')
        self.refresh()

    def cancel_all_workflow(self):
        if not self._workflow_auto_running:
            return
        if self.sequence.state == 'stop_unconfirmed':
            self.message = '停止尚未确认，不能自动恢复；请检查机器人。'
            self.refresh()
            return
        self._workflow_auto_cancel_requested = True
        self._perception_pending=False;self._perception_token=None
        self.record_workflow_event('cancel_requested')
        self._workflow_auto_paused = False
        self._workflow_auto_timer.stop()
        self._execute_after_plan = False
        self._workflow_waiting = False
        self._workflow_confirmation = None
        if self.hand_runner is not None:
            self.hand_runner.stop('取消流程，等待手部停止确认')
        if self.process is not None:
            self.sequence.stop(motion_active=False)
            self.process.terminate()
        if self.runner is not None:
            self.runner.stop('全部流程取消，准备恢复流程初始状态')
        self.message = '正在取消全部流程；停止确认后恢复第一个流程的初始状态。'
        if self.process is None and self.runner is None and self.hand_runner is None:
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
        self._workflow_completed_steps.clear()
        self.task_flow.set_current(0)
        self._apply_workflow_step_settings(0, 0)
        self.message = ('全部流程已取消并恢复到流程初始状态。' if restored else
                        '全部流程已取消，但恢复初始状态失败：' + detail)
        self.record_workflow_event('cancel_finished', restored=restored)
        self.refresh()

    def workflow_status(self):
        if self._program_active and hasattr(self,'phone_program_panel'):return self.phone_program_panel.workflow_text()
        if self.sequence.state == 'stop_unconfirmed':
            return '停止未确认'
        if self._workflow_auto_cancel_requested:
            return '正在取消并恢复'
        if self._workflow_error:
            return '流程失败：' + self._workflow_error
        if self._workflow_auto_paused:
            return '流程已暂停'
        active=self.hand_runner or self.runner
        if active and active.state in ('starting','resuming','pausing','paused','finishing'):
            return {'starting':'执行服务启动/预检中','resuming':'等待继续确认','pausing':'正在暂停',
                    'paused':'已暂停','finishing':'等待执行服务结束'}[active.state]
        if self._perception_pending:return '接近前重新估计目标中'
        if self._workflow_confirmation:
            return '等待人工确认：' + self._workflow_action()['label']
        if self.hand_runner:
            return self.hand_runner.message
        if self._workflow_finished:
            return '流程已完成'
        if self._workflow_waiting:
            return '正在预检或执行机械臂动作'
        return '等待执行下一步'

