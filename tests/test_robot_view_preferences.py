import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt5.QtWidgets import QApplication
from robot_view3d import RobotView3D
from viewer_preferences import load_state, save_main_preferences, save_robot_preferences


class RobotPreferencesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_every_display_selection_survives_recreation_without_restoring_commands(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(RobotView3D, '_load'), patch.object(RobotView3D, '_tick'):
            path = Path(directory)/'state.json'
            first = RobotView3D(preferences_path=path)
            expected = {}
            for key, box in first._display_preferences.items():
                expected[key] = not box.isChecked()
                box.setChecked(expected[key])
            first._restore_scene_check.setChecked(True)
            first.set_simulation_state([0.]*16, {'right':[.3]*6})
            data = load_state(path)
            self.assertEqual(data['robot_view']['selections'], expected)
            second = RobotView3D(preferences_path=path)
            self.assertEqual({k: box.isChecked() for k,box in second._display_preferences.items()}, expected)
            self.assertEqual(second._depth_on['head'], expected['depth.head'])
            self.assertEqual(second._pose_on, {k:expected['pose.'+k] for k in second._pose_checks})
            self.assertEqual(second._skel_on, expected['skeleton'])
            self.assertEqual(second._wrench_on, expected['wrench'])
            self.assertFalse(second._restore_scene_check.isChecked())
            self.assertIsNone(second._simulation_q)
            self.assertEqual(second._sim_hand_controls['right'][0].value(), 0.)
            first.deleteLater();second.deleteLater()

    def test_main_window_save_preserves_robot_choices_even_if_robot_page_never_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.json'
            path.write_text(json.dumps({'widgets':{'existing':123},'geometry':[1,2,3,4]}))
            save_robot_preferences({'skeleton':False}, path)
            self.assertEqual(load_state(path)['widgets'], {'existing':123})
            save_main_preferences({'widgets':{'new':456}}, path)
            self.assertEqual(load_state(path)['robot_view']['selections'], {'skeleton':False})
            self.assertEqual(load_state(path)['widgets'], {'new':456})

    def test_invalid_and_unknown_saved_fields_do_not_enable_actions(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(RobotView3D, '_load'):
            path = Path(directory)/'state.json'
            path.write_text(json.dumps({'robot_view':{'version':1,'selections':{
                'skeleton':'false', 'wrench':False, 'restore_scene':True, 'unknown':True}}}))
            view = RobotView3D(preferences_path=path)
            self.assertTrue(view._skel_check.isChecked())
            self.assertFalse(view._wrench_check.isChecked())
            self.assertFalse(view._restore_scene_check.isChecked())
            view.deleteLater()

    def test_corrupt_preferences_keep_defaults(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(RobotView3D, '_load'):
            path = Path(directory)/'state.json';path.write_text('{broken')
            view = RobotView3D(preferences_path=path)
            self.assertTrue(view._depth_checks['head'].isChecked())
            self.assertFalse(view._reach_checks['l'].isChecked())
            view.deleteLater()


if __name__=='__main__':unittest.main()
