from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.grasp.dependencies')

class ExecutionControllerMixin:
    def read_startup_hands(self):
        from eai_app.domain.robot.commands import HAND_JOINT_ORDER, canonical_hand_joint
        stamps = getattr(self.node, '_grasp_hand_state_received_at', {})
        if any(not 0 <= _deps.time.time()-stamps.get(side, 0) <= 2. for side in ('left','right')):
            raise ValueError('双手反馈尚未就绪或已过期')
        state = self.node.get_robot_state()
        hands = {}
        for side in ('left','right'):
            values = {canonical_hand_joint(name): float(value) for name,value in getattr(state, side+'_hand')}
            vector = [values[name] for name in HAND_JOINT_ORDER]
            from eai_app.domain.grasp.execution import hand_positions
            hands[side] = hand_positions(vector)
        return hands

    def capture_startup_state(self):
        if not self._startup_recording:
            return
        if self._startup_snapshot is None:
            try:
                measured = _deps.read_feedback(_deps.os.environ.get('EAI_HOST_FK_STATE', '/tmp/eai_host_fk_state.json'))
                self._startup_snapshot = _deps.copy.deepcopy(dict(measured, hands={}))
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
        elif _deps.time.monotonic() > self._startup_deadline:
            self._startup_recording = False

    def restore_startup(self):
        if self._program_active:
            self.message='装配程序运行中；请先取消并等待停止。';self.refresh();return
        if self.process or self.runner or self.hand_runner or self._workflow_waiting or self.sequence.state == 'stop_unconfirmed':
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
        if self.use_execution_service and not self.is_simulation and self._unverified_grips:
            raise ValueError('闭合后夹持状态未确认，保持当前位置；确认物体状态后再恢复')
        profile = self.selected_profile()
        measured = self.simulation_backend().feedback() if self.is_simulation else self.feedback()
        from eai_app.planning.restore import prepare_restore
        identifier = _deps.uuid.uuid4().hex
        plan = prepare_restore(profile, measured, startup, identifier)
        if self.use_execution_service:
            from eai_app.planning.validation import certify_plan
            plan=certify_plan(plan,profile,self.runtime_plan_inputs())
        preview = dict(id=identifier, arm_side='both', hand_model=profile['hand_model'],
                       tcp_link='Link7_l / Link7_r', base_frame='base_link',
                       observation_id='restore:'+identifier, goals={'approach':plan['target']})
        self.sequence.reset(preview, profile)
        self.sequence.pending_stage = 'restore'
        self.sequence.plan = dict(plan, accepted_at=_deps.time.monotonic())
        self.sequence.state = 'planned'
        self._restore_hands_goal = _deps.copy.deepcopy(startup.get('hands') or {})
        self._restoring = True
        self._workflow_auto_restoring = bool(auto_restore)
        self._execute_ready_plan()
        if self.runner is None:
            self._restoring = False

    def finish_restore_hands(self):
        hands = self._restore_hands_goal
        if not hands:
            return '双臂已恢复启动状态；启动时未采集到双手反馈，双手恢复已跳过。'
        if self.use_execution_service:
            if not self.is_simulation and self.world.attached:
                raise ValueError('当前仍记录有夹持物体，不能自动恢复手型；请先确认安全放置')
            self._restore_hand_queue=list(hands.items())
            self._start_restore_hand_service()
            return '双臂到位；独立服务正在恢复双手并检查实际反馈。'
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

    def _start_restore_hand_service(self):
        side,values=self._restore_hand_queue[0]
        self._hand_purpose='restore'
        self.hand_runner=self.new_process_runner('hand',side=side,target=values)
        self._workflow_waiting=True

    def change_waist_participation(self, checked):
        if self.runner or self.process or self.reserved:
            return
        self.sequence = _deps.Sequence()
        self.scene_confirm.setChecked(False)
        self.message = ('已允许腰部参与 IK；旧计划已清除，请重新预检。'
                        if checked else '已固定腰部；旧计划已清除，请重新预检。')
        self.refresh()

    def go_perception(self):
        if self.open_perception:
            self.open_perception()
        else:
            self.message = '请打开「分割」页，选择彩色图并点击物体；此处不接受旧多视角面板的近似位姿。'
            self.refresh()

    @property
    def reserved(self):
        if self.is_simulation:
            return self._sim_reserved
        return bool(getattr(self.node, '_grasp_execution_reserved', False))

    def set_reserved(self, value):
        if not value and self._program_active and not self._program_backend_authorized:value=True
        if self.is_simulation:
            self._sim_reserved = value
        else:
            self.node._grasp_execution_reserved = value

    def reserve(self, require_preview=True):
        from eai_app.development.source_guard import assert_sources_current
        assert_sources_current()
        if self._program_active and not self._program_backend_authorized:raise ValueError('装配程序占用控制，请先取消程序')
        if not self.is_simulation and self.other_busy():
            raise ValueError('已有手臂移动/文本队列/遥控，请先结束，不能同时控制')
        if (not self.scene_confirm.isChecked() and not self.is_simulation
                and not self._workflow_auto_restoring):
            raise ValueError('请先确认场景配置与现场一致，且本段路径周围没有未建模障碍')
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
            if self.runner or self.process or self.hand_runner or self._workflow_waiting or self._workflow_auto_running:
                raise ValueError('请先结束当前轨迹或预检')
            self.reserve(require_preview=False)
            stage = 'close' if self.sequence.stage == 'close' else 'open'
            self.feedback()
            profile = self.selected_profile()
            values = profile['open_positions' if stage == 'open' else 'close_positions']
            if self.use_execution_service:
                self._hand_purpose='manual'
                self._manual_hand_operation=(profile['arm_side'],stage)
                self.hand_runner=self.new_process_runner('hand',side=profile['arm_side'],target=values)
                self._workflow_waiting=True
                if self.sequence.preview and self.sequence.stage in ('open','close'):self.sequence.hand_sent()
                self.message='独立服务执行手部验证；等待实际反馈，不自动执行手臂。'
                self.refresh();return
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
            if (not self.runner and not self.process and not self.hand_runner
                    and not self._workflow_waiting and self.sequence.state != 'stop_unconfirmed'):
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
        if self._program_active and not self._program_backend_authorized:return
        try:
            if self.runner or self.process or self.sequence.state == 'stop_unconfirmed':
                raise ValueError('请先结束当前任务或确认停止状态')
            self._execute_after_plan = bool(execute_after)
            self.sequence = _deps.Sequence()
            profile = self.selected_profile(require_calibration=False)
            if not self.observation:
                raise ValueError('请先获取视觉估计结果')
            self._ensure_extrinsic()
            if not self.observation.get('grasps', {}).get('candidates'):
                self.observation['grasps'] = _deps.manual_candidate(
                    self.observation, profile, self.feedback(), [s.value()/100 for s in self.contact_offsets])
                self.populate_candidates()
            result = _deps.make_preview(self.observation, self.candidates.currentIndex(), profile)
            result = self.apply_world_preview(result)
            self.sequence.reset(result, profile)
            if execute_after:
                self.reserve()
            else:
                self.set_reserved(True)
            stage = self.sequence.begin_plan(self.motion_stage.currentData())
            feedback = self.feedback()
            self.launch('plan', stage=stage, start_joints=feedback['q16'],
                        allow_waist=self.waist_participation.isChecked(), **self.runtime_plan_inputs())
        except Exception as exc:
            self._execute_after_plan = False
            if self.process is None and self.runner is None and self.sequence.state != 'stop_unconfirmed':
                self.sequence.fail(str(exc))
                self.set_reserved(False)
            self.message = str(exc)
        self.refresh()

    def execute(self):
        if self._program_active and not self._program_backend_authorized:return
        if self.runner or self.process or self.hand_runner or self._workflow_confirmation or self.sequence.state == 'stop_unconfirmed':
            return
        if self.maybe_reestimate_approach():return
        plan = self.sequence.plan
        valid = (self.sequence.state == 'planned' and plan is not None
                 and plan.get('stage') == self.motion_stage.currentData()
                 and _deps.time.monotonic()-plan['accepted_at'] <= self.sequence.profile['max_plan_age_s'])
        if not valid:
            self.message = '执行请求已收到，正在自动计算目标并预检 IK；通过后才会启动运动。'
            self.execution_feedback.setText(self.message)
            self.plan(execute_after=True)
            return
        self._execute_ready_plan()

    def _execute_ready_plan(self):
        previous_authorized=self._program_backend_authorized
        try:
            if self._program_active and (self.sequence.preview or {}).get('program_goal') and self._program_validator:
                self._program_validator();self._program_backend_authorized=True
            self._startup_recording = False
            self.reserve()
            self.validate_runtime_execution(self.sequence.plan)
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
            if self.use_execution_service:
                self.runner=self.new_process_runner('arm',plan=plan)
                self.message='独立执行服务已启动；界面只监控状态，不驱动轨迹时钟。'
                self.refresh();return
            if self.is_simulation:
                simulation = self.simulation_backend()
                self.runner = _deps.JointRunner(plan, simulation.publish, simulation.feedback, lambda: True)
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
            self.runner = _deps.JointRunner(plan, publish, self.feedback,
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
            if self._program_active:self._program_action_result=('failed',self.message)
        finally:
            self._program_backend_authorized=previous_authorized
        self.refresh()

    def pause_or_resume(self):
        if self.hand_runner is not None:
            try:
                if self.hand_runner.state == 'paused':
                    violation = self.wrench_guard_violation((self.hand_runner.side,))
                    if violation:
                        raise ValueError('六维力仍不满足继续条件：'+violation)
                    self.hand_runner.resume()
                else:
                    self.hand_runner.pause()
                self.message = self.hand_runner.message
            except Exception as exc:
                self.message = str(exc)
            self.refresh()
            return
        if self.runner is None:
            return
        try:
            if self.runner.state == 'paused':
                if self.use_execution_service:self.validate_runtime_execution(self.runner.plan)
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
        from eai_app.development.source_guard import assert_sources_current, source_revision
        assert_sources_current()
        if self.process is not None:
            raise ValueError('已有规划/执行请求')
        self.mode = mode
        self.job_directory = self.root / '.cache' / 'grasp_execution' / _deps.uuid.uuid4().hex
        self.job_directory.mkdir(parents=True)
        request = dict(mode=mode, execution_mode=('simulation' if self.is_simulation else 'hardware'),
                       profile=self.sequence.profile, preview=self.sequence.preview, **extra)
        (self.job_directory / 'request.json').write_text(_deps.json.dumps(request, ensure_ascii=False, allow_nan=False))
        self.process = _deps.QProcess(self)
        environment = _deps.QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONPYCACHEPREFIX', str(self.root / '.cache/bytecode' / source_revision()))
        prefix = _deps.Path(self.sequence.profile['planner_python']).absolute().parent.parent
        for cmeel in sorted(prefix.glob('lib/python*/site-packages/cmeel.prefix')):
            environment.insert('LD_LIBRARY_PATH', str(cmeel / 'lib') + ':' + environment.value('LD_LIBRARY_PATH'))
            for packages in sorted(cmeel.glob('lib/python*/site-packages')):
                environment.insert('PYTHONPATH', str(packages) + ':' + environment.value('PYTHONPATH'))
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(_deps.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.log_tail = ''
        self.started = _deps.time.monotonic()
        self.process.start(self.sequence.profile['planner_python'], [str(self.root / 'compat/host_grasp_plan_worker.py'),
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
        if error == _deps.QProcess.FailedToStart:
            self.finished(-1, _deps.QProcess.CrashExit)

    def finished(self, exit_code, _exit_status):
        if self.process is None:
            return
        self.read_output()
        process, self.process = self.process, None
        process.deleteLater()
        execute_requested = self._execute_after_plan
        try:
            path = self.job_directory / 'result.json'
            result = _deps.json.loads(path.read_text()) if path.is_file() else {
                'ok': False, 'error': 'IK worker 未返回结果: '+self.log_tail}
            if self.mode=='rank_candidates':
                self.accept_candidate_ranking(result);self.set_reserved(False);self.refresh();return
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
                if result.get('solver') == 'bounded_dual_tcp':
                    hold_note += '；已使用带关节限位的联合 IK'
                collision_note=('静态完整几何碰撞复检通过（采样，不含动态障碍）' if result.get('collision_checked')
                                else '未进行碰撞复检')
                self.message = (f"{result['stage']} IK/限位/连续性预检通过，约 {result['times_s'][-1]:.1f} 秒。"
                                f'本段路径确认已保留，可直接点击“执行本段”{hold_note}；{collision_note}。')
        except Exception as exc:
            self.sequence.fail(str(exc))
            self.message = ('执行未启动：' if execute_requested else '') + str(exc)
            self.execution_feedback.setText(self.message)
        self.execution_feedback.setText(self.message)
        if self.sequence.state in ('complete', 'stopped', 'failed'):
            self.set_reserved(False)
        if self._workflow_waiting and self.sequence.state == 'failed':
            self._workflow_waiting = False
            self._workflow_error = self.message
        execute_after = self._execute_after_plan
        self._execute_after_plan = False
        if execute_after and self.sequence.state == 'planned':
            if (self._workflow_auto_running or self._program_active) and self._workflow_auto_paused:
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
            _deps.QTimer.singleShot(0, self._begin_auto_workflow_restore)
        if self._workflow_waiting and self.sequence.state == 'failed':
            self._workflow_waiting = False
            self._workflow_error = self.message
            self._workflow_auto_paused = self._workflow_auto_running
        self.refresh()

    def stop(self):
        if self._program_active and not self._program_backend_authorized:
            self.phone_program_panel.cancel();return
        self._perception_pending=False;self._perception_token=None
        if self._workflow_auto_running and not self._workflow_auto_cancel_requested:
            self.cancel_all_workflow()
            return
        self._execute_after_plan = False
        self._restoring = False
        self._restore_hands_goal = None
        self._wrench_guard_triggered = None
        self._workflow_waiting = False
        self._workflow_confirmation = None
        if self.hand_runner is not None:
            self.hand_runner.stop()
            self.message = self.hand_runner.message
            self.refresh()
            return
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
        if self._perception_pending and _deps.time.monotonic()-self._perception_started>120:
            self._perception_pending=False;self._perception_token=None;self._workflow_waiting=False
            self.set_reserved(False);self._workflow_error='接近前重新估计超时，未执行运动'
            self._workflow_auto_paused=self._workflow_auto_running;self.message=self._workflow_error
        if self.hand_runner:
            if self.hand_runner.state == 'running':
                violation = self.wrench_guard_violation((self.hand_runner.side,))
                if violation:
                    self.hand_runner.stop('六维力保护触发：'+violation, failed=True)
            self.hand_runner.tick()
            hand_state = self.hand_runner.state
            self.message = self.hand_runner.message
            if hand_state in ('complete', 'stopped', 'failed', 'stop_unconfirmed'):
                self.record_workflow_event('hand_finished', state=hand_state,
                    execution_directory=str(getattr(self.hand_runner,'directory','')))
                self.hand_runner = None
                self._workflow_waiting = False
                if self._hand_purpose=='program':
                    self._program_action_result=(hand_state,self.message)
                    if hand_state=='stop_unconfirmed':self.sequence.state=hand_state
                    else:self.set_reserved(False)
                    self.refresh();return
                if self._hand_purpose=='restore':
                    if not self._restoring and hand_state=='complete':
                        hand_state='stopped';self.message='恢复已取消，不继续发送其他手部命令。'
                    if hand_state=='complete':
                        self._restore_hand_queue.pop(0)
                        if self._restore_hand_queue:
                            try:self._start_restore_hand_service()
                            except Exception as exc:hand_state='failed';self.message=str(exc)
                            else:self.refresh();return
                        else:self.sequence.motion_finished(True)
                    if hand_state!='complete':
                        self.sequence.fail(self.message)
                        if hand_state=='stop_unconfirmed':self.sequence.state=hand_state
                    auto=self._workflow_auto_restoring
                    self._restoring=False;self._workflow_auto_restoring=False;self._restore_hands_goal=None
                    if hand_state!='stop_unconfirmed':self.set_reserved(False)
                    if auto:self._finish_auto_workflow_cancel(hand_state=='complete',self.message)
                    else:self.message='双臂和双手已恢复并获得反馈确认。' if hand_state=='complete' else self.message
                    self.refresh();return
                if hand_state == 'stop_unconfirmed':
                    self.sequence.state = 'stop_unconfirmed'
                    self._workflow_error = self.message
                    self._workflow_auto_paused = self._workflow_auto_running
                    self._workflow_auto_cancel_requested = False
                else:
                    self.set_reserved(False)
                    if self._workflow_auto_cancel_requested:
                        self._begin_auto_workflow_restore()
                    elif hand_state == 'complete':
                        if self._hand_purpose=='manual':
                            if self._manual_hand_operation[1]=='open':
                                self.world.detach(self._manual_hand_operation[0],self.feedback());self.pending_attachment=None
                                self._unverified_grips.discard(self._manual_hand_operation[0])
                            self.message='手部位置反馈已到位；夹持结果仍需人工确认。'
                            self.refresh();return
                        if self.use_execution_service:
                            if self._pending_hand_open:
                                self.world.detach(self._pending_hand_side,self.feedback())
                                self._unverified_grips.discard(self._pending_hand_side)
                                self.pending_attachment=None
                            else:
                                self.pending_attachment={'side':self._pending_hand_side,'role':self._pending_hand_role,
                                    'id':self._pending_hand_target,'observation':self._pending_hand_observation}
                        self._advance_task_workflow()
                    else:
                        self._workflow_error = self.message
                        self._workflow_auto_paused = self._workflow_auto_running
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
            if self.runner.state not in ('starting', 'resuming', 'finishing', 'running', 'pausing', 'paused', 'stopping'):
                state, message = self.runner.state, self.runner.message
                if self._program_active:self._program_action_result=(state,message)
                self.record_workflow_event('motion_finished', state=state, result_message=message,
                                           plan=getattr(self.runner, 'plan', None))
                was_auto_restore = self._workflow_auto_restoring
                if self._restoring:
                    if state == 'complete':
                        try:
                            message = self.finish_restore_hands()
                            if self.hand_runner is not None:
                                self.runner=None;self.message=message;self.refresh();return
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
                if not self._workflow_auto_running and not self._program_active:
                    self.scene_confirm.setChecked(False)
                if state == 'complete':
                    self.set_reserved(False)
                if self.sequence.state in ('complete', 'stopped', 'failed'):
                    self.set_reserved(False)
                    self.observation = None  # A physical movement consumes the observation.
                if was_auto_restore:
                    self._finish_auto_workflow_cancel(state == 'complete', message or state)
                elif self._workflow_auto_cancel_requested:
                    _deps.QTimer.singleShot(0, self._begin_auto_workflow_restore)
                elif self._workflow_waiting:
                    self._workflow_waiting = False
                    if state == 'complete':
                        self._advance_task_workflow()
                    else:
                        self.message = '流程步骤未完成：' + (message or state)
                        self._workflow_error = self.message
                        if self._workflow_auto_running:
                            self._workflow_auto_paused = True
        if self.process and self.sequence.state != 'stopped' and _deps.time.monotonic()-self.started > 180:
            self.stop()
            self.message = 'IK 预检超时，已取消'
            self.execution_feedback.setText(self.message)
        self.refresh()

