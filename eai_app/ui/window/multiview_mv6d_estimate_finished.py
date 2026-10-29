"""ui.window.multiview mv6d estimate finished — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_perception_pose_estimation = module_ref('eai_app.perception.pose_estimation')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class MultiviewMv6DEstimateFinishedMixin:
    def _on_mv6d_estimate_finished(self, payload: object) -> None:
        self._mv6d_busy = False
        self._mv6d_lingbot_active = False
        self._mv6d_lingbot_cancel = None
        if hasattr(self, "_update_lingbot_depth_ui"):
            self._update_lingbot_depth_ui()
        self.mv6d_once_btn.setEnabled(True)
        self.mv6d_once_btn.setText("估计一次")
        if hasattr(self, "_on_mv6d_methods_changed"):
            self._on_mv6d_methods_changed()
        if not isinstance(payload, dict):
            finish = getattr(self, '_finish_mv6d_target', None)
            if callable(finish):
                finish({'ok': False, 'error': '估计未返回结果'})
            return
        silent = bool(payload.get("silent"))
        if silent:
            return
        if not payload.get("ok"):
            err = str(payload.get("error") or "未知错误")
            if hasattr(self, '_finish_grasp_reestimate'):
                self._finish_grasp_reestimate(None,err)
            if not silent or payload.get("depth_source") == "lingbot":
                if self._mv6d_last_snap is None:
                    self.mv6d_result_edit.setPlainText(f"失败: {err}")
                self.status_bar.showMessage(f"多视角 6D 失败，保留上次结果: {err[:120]}")
            finish = getattr(self, '_finish_mv6d_target', None)
            if callable(finish):
                finish({'ok': False, 'error': err})
            return
        depth_results = payload.get("lingbot_results") or []
        if depth_results:
            # UI updates stay on the main thread; prefer the head camera cloud.
            depth_result = next((r for r in depth_results if r.get("tag") == "head"), depth_results[0])
            self._on_lingbot_depth_result(depth_result)
        snap = payload.get("snap")
        if not isinstance(snap, _eai_perception_pose_estimation.MultiView6DPoseSnapshot):
            finish = getattr(self, '_finish_mv6d_target', None)
            if callable(finish):
                finish({'ok': False, 'error': '估计快照类型无效'})
            return
        self._mv6d_last_snap = snap
        try:
            self._robot_method_poses = {}
            view = getattr(self, '_robot_view', None)
            if view is not None:
                for key in ('pca', 'foundationpose', 'any6d'):
                    view.set_method_pose(key, None, None)
            self._show_mv6d_on_robot_view(
                snap,
                silent=silent,
                method_poses=payload.get("method_poses"),
            )
        except Exception as exc:
            if not silent:
                self.status_bar.showMessage(f"机器人页未画出 6D：{exc}", 8000)
        from eai_app.perception.grasp_input import observation_from_snapshot, observations_from_methods
        try:
            captured_at = float(payload.get("captured_at", 0))
            base_pose = self._mv6d_pose_in_base(snap)
            observation = observation_from_snapshot(
                snap,
                captured_at,
                f"mv6d:{captured_at:.9f}",
                base_pose=base_pose,
                depth_source=payload.get("depth_source"),
            )
            observations = observations_from_methods(
                snap, captured_at, payload.get("method_poses") or {},
                depth_source=payload.get("depth_source"),
            )
            if not observations:
                method = str(snap.method).lower()
                key = 'foundationpose' if 'foundation' in method else ('any6d' if 'any6d' in method else 'pca')
                observation['observation_id'] += ':' + key
                observations[key] = observation
            for method_key, obs in observations.items():
                obs['pose_method'] = method_key
                obs['provenance'] = _eai_platform_bootstrap.copy.deepcopy(payload.get('provenance') or {})
                head = snap.head_result
                if head is not None and head.source_bgr is not None and head.mask is not None:
                    from grasp_perception import thumbnail
                    ys, xs = _eai_platform_bootstrap.np.where(_eai_platform_bootstrap.np.asarray(head.mask)>0)
                    if len(xs):
                        mh,mw=head.mask.shape[:2];ih,iw=head.source_bgr.shape[:2]
                        roi=[float(xs.min()/mw),float(ys.min()/mh),float((xs.max()+1)/mw),float((ys.max()+1)/mh)]
                        crop=head.source_bgr[int(roi[1]*ih):max(int(roi[1]*ih)+1,int(roi[3]*ih)),int(roi[0]*iw):max(int(roi[0]*iw)+1,int(roi[2]*iw))]
                        obs['provenance']['target_patch']={'roi_normalized':roi,'thumbnail':thumbnail(crop).tolist()}
                obs['quality'] = {'point_count': int(snap.fused_points), 'covariance': None,
                                  'depth_source': payload.get('depth_source')}
            self.grasp_execution_panel.set_method_observations(observations)
            payload['observations'] = observations
            if hasattr(self, '_finish_grasp_reestimate'):
                self._finish_grasp_reestimate(observations,target_id=payload.get('requested_target_id'))
            target = getattr(self, '_mv6d_active_target', None)
            if target is not None:
                payload['target'] = dict(target)
                self._mv6d_target_results[target['id']] = _eai_platform_bootstrap.copy.deepcopy(payload)
            from eai_app.infrastructure.observation_store import save_cache
            save_cache(self._mv6d_cache_path(), {
                'snap': snap, 'captured_at': captured_at,
                'method_poses': getattr(self, '_robot_method_poses', {}),
                'observations': observations,
                'sam3_points': getattr(self, '_robot_sam3_points', None),
                'lingbot_results': payload.get('lingbot_results') or [],
                'robot_lingbot_depth': getattr(self, '_robot_lingbot_depth', None),
                'depth_source': payload.get('depth_source'),
                'targets': _eai_platform_bootstrap.copy.deepcopy(getattr(self, '_mv6d_targets', [])),
                'target_results': _eai_platform_bootstrap.copy.deepcopy(list(getattr(self, '_mv6d_target_results', {}).values())),
            })
        except Exception as exc:
            if hasattr(self, '_finish_grasp_reestimate'):
                self._finish_grasp_reestimate(None,str(exc))
            self.grasp_execution_panel.message = f"视觉6D输入转换失败: {exc}"
            self.grasp_execution_panel.refresh()

        # 先用仿真物体自动校准 GT，再刷新误差显示（避免手填 GT 导致「假 800mm」）
        try:
            xyz_est, _ = self._mv6d_xyz_in_base_link(snap)
            if not silent:
                self._mv6d_auto_gt_from_sim(xyz_est)
        except Exception:
            pass
        self._refresh_mv6d_result_text(snap)
        try:
            self._apply_mv6d_overlays(snap)
        except Exception as exc:
            if not silent:
                self.status_bar.showMessage(f"多视角 6D 叠加失败: {exc}")
        elapsed = float(payload.get("elapsed_s") or 0.0)
        filled = False
        if not silent:
            try:
                self._adopt_mv6d_segment_target(snap)
            except Exception as exc:
                self.status_bar.showMessage(f"分割位姿未写入: {exc}")
        if (
            not silent
            and getattr(self, "mv6d_fill_cmd_check", None) is not None
            and self.mv6d_fill_cmd_check.isChecked()
        ):
            try:
                filled = self._fill_arm_text_from_mv6d(snap)
            except Exception as exc:
                self.status_bar.showMessage(f"多视角 6D 回填文本指挥失败: {exc}")
        if not silent:
            side_name = _eai_perception_pose_estimation.arm_side_label(self._mv6d_cmd_arm_side())
            extra = f"；已回填{side_name}绝对目标，点「执行」移动" if filled else ""
            self.status_bar.showMessage(
                f"多视角 6D 完成 ({snap.method}, {elapsed:.2f}s, "
                f"pts={snap.fused_points}){extra}"
            )
        finish = getattr(self, '_finish_mv6d_target', None)
        if callable(finish):
            finish(payload)

    def _mv6d_cache_path(self):
        return _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, '.cache', 'mv6d_last_success.npz')

