"""Offscreen UI regression tests; all robot I/O is replaced with a fake node."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

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
