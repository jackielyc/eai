from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.robot.dependencies')

class InteractionMixin:
    def _init_display_preferences(self, path):
        from eai_app.infrastructure.viewer_preferences import DEFAULT_PATH, load_state
        self._preferences_path = DEFAULT_PATH if path is None else path
        # Explicit whitelist: display selections only, never hand commands or
        # the operator's transient confirmation that a real motion is safe.
        self._display_preferences = {
            **{'depth.'+key: box for key, box in self._depth_checks.items()},
            **{'pose.'+key: box for key, box in self._pose_checks.items()},
            **{'reachable.'+key: box for key, box in self._reach_checks.items()},
            'skeleton': self._skel_check, 'wrist_accessories': self._wrist_accessories_check,
            'sam3': self._sam3_check, 'lingbot': self._lingbot_check,
            'wrench': self._wrench_check, 'targets': self._target_poses_check,
            'collision_scene': self._collision_scene_check,
        }
        saved = load_state(self._preferences_path).get('robot_view', {})
        values = saved.get('selections', {}) if isinstance(saved, dict) and saved.get('version') == 1 else {}
        if isinstance(values, dict):
            for key, box in self._display_preferences.items():
                if type(values.get(key)) is bool:
                    box.setChecked(values[key])  # Update both the checkbox and layer state before GL loading.
        for box in self._display_preferences.values():
            box.toggled.connect(self._save_display_preferences)

    def _save_display_preferences(self, *_):
        from eai_app.infrastructure.viewer_preferences import save_robot_preferences
        try:
            save_robot_preferences({key: box.isChecked() for key, box in self._display_preferences.items()},
                                   self._preferences_path)
        except (OSError, ValueError) as exc:
            self._status.setText('机器人页显示选项未保存：'+str(exc))

    def _refresh_reach_visibility(self, *_):
        for side, item in self._reach_items.items():
            item['enabled'] = self._reach_checks[side].isChecked() and self._reach_waist is not None
        if self._gl is not None:
            self._gl.update()

    def _update_reachable_space(self, q16, age):
        if not any(box.isChecked() for box in self._reach_checks.values()):
            return
        if q16 is None or age < 0 or age > 2 or not _deps.np.isfinite(q16).all():
            for item in self._reach_items.values():
                item['enabled'] = False
            self._reach_note.setText('可达空间等待新鲜关节反馈：请启动 FK 反馈。')
            return
        waist = _deps.np.asarray(q16[:2])
        job = self._reach_job
        if job is not None:
            if not job['done']:
                return
            self._reach_job = None
            if job.get('error'):
                self._reach_note.setText('可达空间计算失败：' + job['error'])
                return
            if _deps.np.allclose(waist, job['waist'], atol=.005, rtol=0):
                for side, points in job['points'].items():
                    packed = _deps.np.empty((len(points), 6), dtype=_deps.np.float32)
                    packed[:, :3] = points
                    packed[:, 3:] = (0.1, .85, 1.) if side == 'l' else (1., .55, .15)
                    item = self._reach_items[side]
                    with item['lock']:
                        item['pending'] = packed
                self._reach_waist = waist.copy()
                self._reach_note.setText('左右臂各 6000 个 Link7 可达采样点；固定当前腰部姿态，不含碰撞、手指接触点及指定朝向约束。')
        if self._reach_waist is not None and _deps.np.allclose(waist, self._reach_waist, atol=.005, rtol=0):
            self._refresh_reach_visibility()
            return
        self._reach_waist = None
        self._refresh_reach_visibility()
        self._reach_note.setText('正在计算当前腰部姿态下的可达空间…')
        job = {'done': False, 'waist': waist.copy()}
        self._reach_job = job
        values = list(q16)
        def compute():
            try:
                job['points'] = _deps.sample_reachable_positions(_deps._resolve_urdf(), values)
            except Exception as exc:
                job['error'] = str(exc)
            finally:
                job['done'] = True
        _deps.threading.Thread(target=compute, daemon=True).start()

    def set_grasp_preview(self, preview) -> None:
        for stage in ('pregrasp', 'approach', 'lift'):
            goal = (preview or {}).get('goals', {}).get(stage)
            if goal is None:
                self.set_method_pose('grasp_' + stage, None, None)
            else:
                transform = _deps.np.asarray(goal, dtype=float)
                self.set_method_pose('grasp_' + stage, transform[:3, 3], transform[:3, :3],
                                     extents=(.012, .012, .012))
        if preview:
            prefix = '仅诊断：目标越界，不能执行。' if preview.get('diagnostic_only') else '抓取目标预览，尚未进行轨迹碰撞检查。'
            self._status.setText(prefix + ' 紫=预抓取，绿=接近，黄=抬起；可分别取消勾选。')

    def set_simulation_state(self, joints, hands):
        self._simulation_q = None if joints is None else list(joints)
        self._simulation_hands = dict(hands)
        for side, boxes in self._sim_hand_controls.items():
            for box, value in zip(boxes, hands.get(side, [0.] * 6)):
                box.blockSignals(True)
                box.setValue(value)
                box.setEnabled(joints is not None and self._simulation_manual_allowed())
                box.blockSignals(False)
        self._simulation_label.setText('仿真执行：模型姿态，不是真机反馈；无物体接触仿真'
                                       if joints is not None else '机器人姿态来源：真机反馈')
        self._tick()

    def _simulation_manual_allowed(self):
        callback=getattr(self,'simulation_controls_allowed',None)
        return bool(callback()) if callback else True

    def _change_simulation_hand(self, side):
        if self._simulation_q is not None and self.simulation_hand_callback and self._simulation_manual_allowed():
            self.simulation_hand_callback(side, [box.value() for box in self._sim_hand_controls[side]])

    def _grasp_action(self, action):
        if self.grasp_action_callback:
            self._grasp_action_note.setText(self.grasp_action_callback(action))

