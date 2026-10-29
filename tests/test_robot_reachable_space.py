import unittest
from types import SimpleNamespace
import numpy as np
from robot_view3d import (sample_reachable_positions, _resolve_urdf,
                          _Kinematics, RobotView3D, JOINT_NAMES)


class ReachableSpaceTests(unittest.TestCase):
    def test_samples_match_fk_and_preserve_waist(self):
        path = _resolve_urdf()
        q = [0.] * 16
        q[0], q[1] = .1, .15
        points = sample_reachable_positions(path, q, 40)
        kin = _Kinematics(path)
        rng = np.random.default_rng(17)
        for side in ('l', 'r'):
            indices = [kin._q_idx[f'joint{i}_{side}'] for i in range(1, 8)]
            angles = rng.uniform(kin.model.lowerPositionLimit[indices],
                                 kin.model.upperPositionLimit[indices], (40, 7))
            self.assertEqual(points[side].shape, (40, 3))
            for i in (0, 19, 39):
                values = q.copy()
                for j, angle in enumerate(angles[i], 1):
                    values[JOINT_NAMES.index(f'joint{j}_{side}')] = angle
                expected = kin.link_poses(values)[f'Link7_{side}'][:3, 3]
                np.testing.assert_allclose(points[side][i], expected)

    def test_visibility_and_stale_feedback_hide_points(self):
        view = SimpleNamespace(
            _reach_items={'l': {}, 'r': {}}, _reach_waist=np.zeros(2), _gl=None,
            _reach_checks={'l': SimpleNamespace(isChecked=lambda: True),
                           'r': SimpleNamespace(isChecked=lambda: False)},
            _reach_note=SimpleNamespace(setText=lambda text: None),
        )
        RobotView3D._refresh_reach_visibility(view)
        self.assertTrue(view._reach_items['l']['enabled'])
        self.assertFalse(view._reach_items['r']['enabled'])
        RobotView3D._update_reachable_space(view, [0.]*16, 5.)
        self.assertFalse(view._reach_items['l']['enabled'])


if __name__ == '__main__':
    unittest.main()
