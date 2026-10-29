import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from robot_view3d import RuiyanHandModel, RobotView3D


class HandModelTests(unittest.TestCase):
    def setUp(self):
        self.preferences = tempfile.TemporaryDirectory()
        self.preferences_path = Path(self.preferences.name)/'state.json'

    def tearDown(self):
        self.preferences.cleanup()
    @classmethod
    def setUpClass(cls):
        cls.model = RuiyanHandModel()
        cls.app = QApplication.instance() or QApplication([])

    def test_complete_meshes_follow_wrist_and_independent_fingers(self):
        arms = {'Link7_l': np.eye(4), 'Link7_r': np.eye(4)}
        opened = self.model.poses(arms, {})
        closed = self.model.poses(arms, {'right': {'index': .6}})
        self.assertEqual(len({v['link'] for v in self.model.visuals}), 28)
        self.assertTrue({v['link'] for v in self.model.visuals}.issubset(opened))
        for name in opened:
            self.assertTrue(np.isfinite(opened[name]).all())
            if name.startswith('hand2_link_2_'):
                self.assertFalse(np.allclose(opened[name], closed[name]))
            else:
                np.testing.assert_allclose(opened[name], closed[name])
        arms['Link7_l'][:3, 3] = [.2, .3, .4]
        moved = self.model.poses(arms, {})
        np.testing.assert_allclose(moved['hand1_link_0'][:3, 3] - opened['hand1_link_0'][:3, 3], [.2,.3,.4])
        for link in ('wrist_connector_0', 'wrist_camera_0'):
            np.testing.assert_allclose(moved[link][:3, 3] - opened[link][:3, 3], [.2,.3,.4])
        for link in ('wrist_connector_1', 'wrist_camera_1'):
            np.testing.assert_allclose(moved[link], opened[link])

    def test_hand_controls_only_callback_in_simulation(self):
        with patch.object(RobotView3D, '_load'), patch.object(RobotView3D, '_tick'):
            view = RobotView3D(preferences_path=self.preferences_path)
            calls = []
            view.simulation_hand_callback = lambda side, values: calls.append((side, values))
            self.assertFalse(view._sim_hand_controls['right'][2].isEnabled())
            view.set_simulation_state([0.] * 16, {})
            self.assertFalse(calls)
            view._sim_hand_controls['right'][2].setValue(.5)
            self.assertEqual(calls, [('right', [0.,0.,.5,0.,0.,0.])])
            view.set_simulation_state(None, {})
            self.assertFalse(view._sim_hand_controls['right'][2].isEnabled())
            self.assertEqual(len(calls), 1)
            view.deleteLater()

    def test_multi_target_pose_layer_can_be_toggled(self):
        with patch.object(RobotView3D, '_load'), patch.object(RobotView3D, '_tick'):
            view=RobotView3D(preferences_path=self.preferences_path);view._target_pose_item=view._make_pose6d()
            view.set_target_poses([{'xyz':[.3,.1,.8],'rotation':np.eye(3),
                                    'extents':[.1,.2,.3]}])
            self.assertTrue(view._target_pose_item['enabled'])
            self.assertGreater(len(view._target_pose_item['pending']),0)
            view._target_poses_check.setChecked(False)
            self.assertFalse(view._target_pose_item['enabled'])
            view.deleteLater()


if __name__ == '__main__': unittest.main()
