"""Offscreen UI regression tests; all robot I/O is replaced with a fake node."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication

from grasp_execution_ui import GraspExecutionPanel
from test_grasp_execution import example


class FakeNode:
    def __init__(self):
        self.hand_commands = []
        self.arm_commands = []
        self._grasp_execution_reserved = False
    def is_arm_enabled(self): return True
    def is_hand_enabled(self): return True
    def apply_hand_joint_positions(self, side, positions):
        assert self._grasp_hand_command_authorized
        self.hand_commands.append((side, list(positions)))


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.node = FakeNode()
        self.panel = GraspExecutionPanel(self.node, busy=lambda: False)
        self.panel.execution_mode.setCurrentIndex(1)
        self.profile, self.obs = example()
        path = Path(self.tmp.name)/'profile.json'
        path.write_text(json.dumps(self.profile))
        self.panel.profile_path.setText(str(path))
        self.panel.feedback = lambda: {'q16': [0]*16}

    def tearDown(self):
        self.panel.watchdog.stop()
        self.panel.deleteLater()
        self.tmp.cleanup()

    def test_empty_panel_explains_blockers_and_allows_setup(self):
        self.panel.profile_path.setText('/tmp/nonexistent-grasp-profile-for-test.json')
        self.panel.check_button.click()
        self.assertIn('抓取配置不存在', self.panel.readiness_label.text())
        self.assertIn('未收到视觉6D估计', self.panel.readiness_label.text())
        self.assertTrue(self.panel.browse_profile_button.isEnabled())
        self.assertTrue(self.panel.perception_button.isEnabled())
        self.assertFalse(self.panel.execute_button.isEnabled())
        self.assertFalse(self.node.hand_commands)
        calls = []
        self.panel.open_perception = lambda: calls.append('perception')
        self.panel.perception_button.click()
        self.assertEqual(calls, ['perception'])

    def test_feedback_button_requests_safe_fk_start(self):
        calls = []
        self.panel.ensure_feedback = lambda: calls.append('start')
        self.panel.feedback_button.click()
        self.assertEqual(calls, ['start'])
        self.assertIn('已请求启动 FK', self.panel.message)

    def test_simulation_hand_execute_pause_stop_never_call_hardware(self):
        from grasp_simulation import GraspSimulation
        from robot_view3d import _resolve_urdf
        frames = []
        self.panel.execution_mode.setCurrentIndex(0)
        self.profile['urdf_path'] = _resolve_urdf()
        Path(self.panel.profile_path.text()).write_text(json.dumps(self.profile))
        sim = GraspSimulation(self.profile, [0.] * 16, lambda q, h: frames.append((q, h)))
        self.panel.simulation = sim
        self.panel.feedback = sim.feedback
        self.node.is_arm_enabled = lambda: False
        self.node.is_hand_enabled = lambda: False
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.panel.scene_confirm.setChecked(True)
        self.panel.hand_button.click()
        self.assertEqual(len(frames), 2)  # Preview initializes the model, then the hand command updates it.
        self.assertEqual(frames[-1][1]['right'], self.profile['open_positions'])
        self.assertFalse(self.node.hand_commands)
        self.assertFalse(self.node._grasp_execution_reserved)
        self.panel.confirm_button.click()
        seq = self.panel.sequence
        seq.begin_plan()
        end = [0.] * 16
        end[9] = .01
        target = sim.kin.link_poses(end)['Link7_r'].tolist()
        result = dict(ok=True, preview_id=seq.preview['id'], stage='pregrasp',
                      backend='host_ik_manual', joint_limits_checked=True, ik_checked=True,
                      joint_waypoints=[[0.] * 16, end], times_s=[0, 1],
                      urdf_sha256=sim.model_hash, arm_side='right', target=target)
        seq.accept_plan(result)
        self.panel.refresh()
        self.panel.execute_button.click()
        self.assertIsNotNone(self.panel.runner)
        self.assertFalse(self.panel.execution_mode.isEnabled())
        self.panel.runner.started -= .2
        self.panel.monitor()
        self.assertGreater(sim.q[9], 0.)
        self.assertLess(sim.q[9], .01)
        self.panel.pause_button.click()
        for _ in range(3): self.panel.monitor()
        self.assertEqual(self.panel.runner.state, 'paused')
        self.panel.pause_button.click()
        self.assertEqual(self.panel.runner.state, 'running')
        self.panel.stop_button.click()
        for _ in range(3): self.panel.monitor()
        self.assertIsNone(self.panel.runner)
        self.assertFalse(self.node.hand_commands)
        self.assertFalse(self.node.arm_commands)
        self.assertGreater(len(frames), 1)

    def test_successful_preflight_keeps_confirmation_and_allows_execute_click(self):
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.panel.scene_confirm.setChecked(True)
        seq = self.panel.sequence
        seq.hand_sent()
        seq.confirm_hand()
        seq.begin_plan()
        result = dict(ok=True, preview_id=seq.preview['id'], stage='pregrasp',
                      backend='host_ik_manual', joint_limits_checked=True,
                      ik_checked=True, times_s=[0, 1])
        self.panel.job_directory = Path(self.tmp.name)
        (self.panel.job_directory / 'result.json').write_text(json.dumps(result))
        self.panel.process = SimpleNamespace(deleteLater=lambda: None)
        with patch.object(self.panel, 'read_output'):
            self.panel.finished(0, 0)
        self.assertTrue(self.panel.scene_confirm.isChecked())
        self.assertTrue(self.panel.execute_button.isEnabled())
        with patch('grasp_execution_ui.JointRunner') as runner, \
                patch.object(self.node, '_enter_model_mode_for_motion', create=True), \
                patch.object(self.node, '_release_ik_for_joint_move', create=True):
            self.panel.execute_button.click()
            runner.assert_called_once()
            self.assertEqual(seq.state, 'executing')
        self.panel.runner = None

    def test_validity_saved_and_expiry_visible(self):
        self.assertEqual(self.panel.plan_validity.value(), 600)
        self.panel.plan_validity.setValue(900)
        self.panel.save_validity_button.click()
        self.assertEqual(json.loads(Path(self.panel.profile_path.text()).read_text())['max_plan_age_s'], 900)
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.panel.sequence.state = 'planned'
        self.panel.sequence.plan = {'accepted_at': 100}
        with patch('grasp_execution_ui.time.monotonic', return_value=101):
            self.panel.refresh()
        self.assertIn('14:59', self.panel.plan_expiry_label.text())
        self.assertTrue(self.panel.execute_button.isEnabled())
        with patch('grasp_execution_ui.time.monotonic', return_value=1001):
            self.panel.refresh()
        self.assertIn('过期', self.panel.plan_expiry_label.text())
        self.assertFalse(self.panel.execute_button.isEnabled())

    def test_source_switch_discards_preview_and_uses_current_batch(self):
        first = copy.deepcopy(self.obs)
        second = copy.deepcopy(self.obs)
        first['observation_id'] = 'pca-result'
        second['observation_id'] = 'any6d-result'
        second['grasps'] = {}
        self.panel.set_method_observations({'pca': first, 'any6d': second})
        self.panel.preview()
        self.assertIsNotNone(self.panel.sequence.preview)
        self.panel.source_combo.setCurrentIndex(2)
        self.assertEqual(self.panel.observation['observation_id'], 'any6d-result')
        self.assertIsNone(self.panel.sequence.preview)
        self.assertEqual(self.panel.candidates.count(), 0)
        self.assertFalse(self.panel.source_combo.model().item(1).isEnabled())
        self.panel.set_method_observations({'pca': first})
        self.assertEqual(self.panel.source_combo.currentData(), 'pca')
        self.assertFalse(self.panel.source_combo.model().item(2).isEnabled())
        self.assertFalse(self.node.hand_commands)

    def test_shape_click_without_observation_displays_reason(self):
        self.panel.suggest_shape_button.click()
        self.assertIn('请先完成视觉6D估计', self.panel.shape_result_dialog.text())
        self.assertIn('手型估计失败', self.panel.report.toPlainText())
        self.panel.shape_result_dialog.close()

    def test_shape_click_with_execution_blocker_still_shows_geometry(self):
        self.obs['object_pose']['extents_m'] = [.04, .05, .06]
        self.obs['execution_blockers'] = ['缺少外参']
        self.panel.set_observation(self.obs)
        self.panel.suggest_shape_button.click()
        self.assertIn('三指捏取', self.panel.shape_result_dialog.text())
        self.assertIn('缺少外参', self.panel.shape_result_dialog.text())
        self.assertIsNone(self.panel.shape_override)
        self.assertFalse(self.node.hand_commands)
        self.panel.shape_result_dialog.close()

    def test_calibration_button_opens_operator_owned_setup(self):
        calls = []
        self.panel.open_calibration = lambda: calls.append('calibration')
        self.panel.calibration_button.click()
        self.assertEqual(calls, ['calibration'])

    def test_pose_arrival_and_preview_send_no_commands(self):
        self.panel.set_observation(self.obs)
        self.panel.preview_button.click()
        self.assertIsNotNone(self.panel.sequence.preview)
        self.assertFalse(self.node.hand_commands)
        self.assertFalse(self.node.arm_commands)
        self.assertFalse(self.panel.execute_button.isEnabled())

    def test_robot_preview_receives_success_and_blocked_workspace_geometry(self):
        displayed = []
        self.panel.robot_preview_callback = displayed.append
        self.panel.set_observation(self.obs)
        self.panel.preview_button.click()
        self.assertIn('pregrasp', displayed[-1]['goals'])
        self.assertIsNotNone(self.panel.sequence.preview)
        self.profile['workspace_m'] = [[0, 0, 0], [.01, .01, .01]]
        Path(self.panel.profile_path.text()).write_text(json.dumps(self.profile))
        self.panel.preview_button.click()
        self.assertTrue(displayed[-1]['diagnostic_only'])
        self.assertIsNone(self.panel.sequence.preview)
        self.assertFalse(self.panel.hand_button.isEnabled())
        self.assertIn('XYZ=', self.panel.message)
        self.assertFalse(self.node.hand_commands)

    def test_shape_estimation_applies_ruiyan_factory_mapping_without_motion(self):
        self.profile['hand_model'] = 'ruiyan_6dof_v1'
        self.profile['open_positions'] = [0.0] * 6
        self.profile['close_positions'] = [0.9, 0.9, 1.0, 1.0, 1.0, 1.0]
        self.obs['grasps']['gripper'] = 'ruiyan_6dof_v1'
        self.obs['object_pose']['extents_m'] = [.04, .05, .06]
        path = Path(self.panel.profile_path.text())
        path.write_text(json.dumps(self.profile))

        self.panel.set_observation(self.obs)

        selected = self.panel.selected_profile()
        np.testing.assert_allclose(selected['open_positions'], [.45, .45, .5, .5, 0., 0.])
        np.testing.assert_allclose(selected['close_positions'], [.525, .525, 7/12, 7/12, 0., 0.])
        self.assertIn('已用于本次抓取', self.panel.hand_shape_label.text())
        self.assertFalse(self.node.hand_commands)
        self.assertFalse(self.node.arm_commands)

    def test_open_is_explicit_and_requires_manual_completion_confirmation(self):
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.panel.hand_button.click()
        self.assertFalse(self.node.hand_commands)
        self.panel.scene_confirm.setChecked(True)
        self.panel.hand_button.click()
        self.assertEqual(self.node.hand_commands, [('right', self.profile['open_positions'])])
        self.assertEqual(self.panel.sequence.state, 'await_open')
        self.assertFalse(self.panel.plan_button.isEnabled())
        self.panel.confirm_button.click()
        self.assertEqual(self.panel.sequence.state, 'opened')
        self.assertTrue(self.panel.plan_button.isEnabled())
        self.assertFalse(self.node.arm_commands)
        self.panel.stop_button.click()
        self.assertEqual(self.panel.sequence.state, 'stopped')
        self.assertFalse(self.node._grasp_execution_reserved)
        self.assertEqual(len(self.node.hand_commands), 1)  # Stop never auto-opens/closes.

    def test_new_observation_cannot_replace_locked_target(self):
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.panel.scene_confirm.setChecked(True)
        self.panel.hand()
        other = dict(self.obs, observation_id='other')
        self.panel.set_observation(other)
        self.assertEqual(self.panel.sequence.preview['observation_id'], 'test')
        self.panel.stop()

    def test_legacy_segmentation_no_longer_schedules_auto_motion(self):
        tree = ast.parse(Path(__file__).with_name('show_camera_topics.py').read_text())
        for name in ('_store_segment_target', '_try_auto_move_to_segment_target'):
            method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n.func.attr for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            self.assertNotIn('singleShot', calls)
            self.assertNotIn('request_slow_move_to_goal', calls)
            self.assertNotIn('_start_sim_arm_move', calls)

    def test_reserved_control_rejects_legacy_joint_and_hand_publishers(self):
        tree = ast.parse(Path(__file__).with_name('show_camera_topics.py').read_text())
        for name in ('_publish_wbc_joint_target', '_publish_hand_command'):
            method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
            scope = {'List': list, 'JointState': object}
            exec(compile(ast.Module(body=[method], type_ignores=[]), '<node method>', 'exec'), scope)
            fake = SimpleNamespace(_grasp_execution_reserved=True,
                                   get_logger=lambda: SimpleNamespace(warn=lambda _: None))
            if name == '_publish_wbc_joint_target':
                self.assertIsNone(scope[name](fake, [0]*16))
            else:
                self.assertFalse(scope[name](fake, 'right', object()))


if __name__ == '__main__':
    unittest.main()
