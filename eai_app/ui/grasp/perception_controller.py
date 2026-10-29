from __future__ import annotations
from eai_app.infrastructure.module_ref import module_ref
_deps=module_ref('eai_app.ui.grasp.dependencies')

class PerceptionControllerMixin:
    def show_shape_result(self):
        self.suggest_shape()
        self.shape_result_dialog = _deps.QMessageBox(self)
        self.shape_result_dialog.setWindowTitle('物体手型估计结果')
        self.shape_result_dialog.setText(self.hand_shape_label.text())
        self.shape_result_dialog.setStandardButtons(_deps.QMessageBox.Ok)
        self.shape_result_dialog.setModal(False)
        self.shape_result_dialog.show()

    def suggest_shape(self):
        from eai_app.domain.grasp.hand_shape import calibrated_preset, suggest_hand_shape
        if self.reserved or self.process or self.runner:
            return
        self.sequence = _deps.Sequence()
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
                profile = _deps.load_profile(profile_path, require_calibration=not self.is_simulation)
                opened, closed = calibrated_preset(profile, item)
                self.shape_override = (self.observation['observation_id'], profile_path, opened, closed)
                self.sequence = _deps.Sequence()
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
        from eai_app.domain.grasp.hand_shape import calibrated_preset
        try:
            if self.reserved or self.process or not self.hand_shape_suggestion:
                raise ValueError('请先估计手型并结束当前动作')
            if self.observation.get('execution_blockers'):
                raise ValueError('；'.join(self.observation['execution_blockers']))
            profile = self.selected_profile(require_calibration=True)
            opened, closed = calibrated_preset(profile, self.hand_shape_suggestion)
            self.shape_override = (self.observation['observation_id'], self.profile_path.text().strip(), opened, closed)
            self.sequence = _deps.Sequence()
            self.message = '已选择该物体的标定手型；重新预览后生效，没有发送开合命令。'
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def _ensure_extrinsic(self):
        if self.fill_extrinsic and self.observation:
            self.observation = self.fill_extrinsic(self.observation)

    def invalidate_preview(self, *_):
        if not self.reserved and self.process is None:
            self.sequence = _deps.Sequence()
            self.refresh()

    def offset_changed(self, *_):
        if not self.reserved and self.observation and self.observation.get('grasps', {}).get('model') == 'manual_current_hand_orientation':
            self.observation.pop('grasps', None)
            self.candidates.clear()
            self.sequence = _deps.Sequence()
            self.message = '接触偏移已改变，请重新生成候选并预览'
            self.refresh()

    def set_observation(self, observation):
        if self.reserved or self.process is not None:
            self.message = '抓取进行中：新观测未替换已锁定目标；需要更换时先停止'
            self.refresh()
            return
        self.observation = {k: v for k, v in observation.items() if k not in ('mask', 'image', 'K')}
        self._ensure_extrinsic()
        self.sequence = _deps.Sequence()
        self.scene_confirm.setChecked(False)
        self.populate_candidates()
        self.shape_override = None
        self.suggest_shape()
        self.refresh()

    def set_target_observations(self, targets):
        if self.reserved or self.process or self.runner:
            return
        selected = str(self.grasp_target_combo.currentData() or '')
        self.target_observations = _deps.copy.deepcopy(targets)
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
            self.sequence = _deps.Sequence()
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

    def set_method_observations(self, observations):
        if self.reserved or self.process or self.runner:
            return
        selected = self.source_combo.currentData()
        self.method_observations = _deps.copy.deepcopy(observations)
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
            self.sequence = _deps.Sequence()
            self.candidates.clear()
            self.message = '本轮没有可用的方法结果，请先进行视觉6D估计。'
            self.refresh()
            return
        self.set_observation(_deps.copy.deepcopy(observation))
        self.check_readiness()

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
            self.observation['grasps'] = _deps.manual_candidate(
                self.observation, profile, self.feedback(), [s.value()/100 for s in self.contact_offsets])
            self.sequence = _deps.Sequence()
            self.populate_candidates()
            extra = '' if profile.get('calibrated') else ' 配置未标定，只能预览，不能执行。'
            self.message = '已生成候选：人工接触点+当前手朝向。请预览并核对接触位置，尚未做自动避障。' + extra
        except Exception as exc:
            self.message = str(exc)
        self.refresh()

    def import_candidates(self):
        if not self.observation or self.reserved or self.process:
            return
        path, _ = _deps.QFileDialog.getOpenFileName(self, '导入本次观测的手型抓取候选', '', 'JSON (*.json)')
        if not path:
            return
        try:
            data = _deps.json.loads(_deps.Path(path).read_text())
            if data.get('observation_id') != self.observation.get('observation_id'):
                raise ValueError('候选不属于当前观测，不能复用旧坐标')
            if not data.get('ok') or not data.get('candidates') or not data.get('gripper'):
                raise ValueError('候选文件需要 ok、gripper 和 candidates')
            from eai_app.domain.geometry.transforms import validate_transform
            for candidate in data['candidates']:
                validate_transform(candidate['T_camera_grasp'])
                if not _deps.np.isfinite(float(candidate.get('model_score', 0))):
                    raise ValueError('候选分数无效')
            self.observation['grasps'] = data
            self.sequence = _deps.Sequence()
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
                self.observation['grasps'] = _deps.manual_candidate(
                    self.observation, profile, self.feedback(), [s.value()/100 for s in self.contact_offsets])
                self.populate_candidates()
            result = _deps.make_preview(self.observation, self.candidates.currentIndex(), profile)
            result = self.apply_world_preview(result)
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

