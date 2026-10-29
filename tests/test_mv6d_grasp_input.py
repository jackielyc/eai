import unittest
from types import SimpleNamespace

import numpy as np

from mv6d_grasp_input import observation_from_snapshot, observations_from_methods


class Mv6dInputTests(unittest.TestCase):
    def test_methods_keep_separate_geometry_and_identity(self):
        poses = {'pca': ([.1, .2, .3], np.eye(3), None, [.02, .03, .04]),
                 'any6d': ([.4, .5, .6], np.eye(3), None, [.05, .06, .07])}
        result = observations_from_methods(self.snapshot(), 1., poses, 'lingbot')
        self.assertEqual(set(result), {'pca', 'any6d'})
        self.assertNotEqual(result['pca']['observation_id'], result['any6d']['observation_id'])
        np.testing.assert_allclose(np.asarray(result['any6d']['T_base_object'])[:3, 3], [.4, .5, .6])
        self.assertEqual(result['any6d']['object_pose']['extents_m'], [.05, .06, .07])
        self.assertFalse(result['any6d']['execution_blockers'])

    def snapshot(self, frame='camera_frame', tf_ok=False):
        return SimpleNamespace(
            frame_id=frame,
            tf_ok=tf_ok,
            rotation_matrix=np.eye(3),
            position_xyz=(.1, .2, .3),
            obb_extents=(.02, .03, .04),
            method='sam3+pca',
            notes='',
            head_result=None,
            left_wrist_result=None,
            right_wrist_result=None,
            wrist_result=None,
        )

    def test_camera_frame_is_rejected_without_verified_transform(self):
        result = observation_from_snapshot(self.snapshot(), 1., 'obs')
        self.assertNotIn('T_base_camera', result)
        self.assertTrue(result['execution_blockers'])

    def test_ui_resolved_base_transform_is_preserved(self):
        result = observation_from_snapshot(
            self.snapshot(), 1., 'obs',
            base_pose=(np.array([.5, -.2, .8]), np.eye(3), None, (.02, .03, .04)),
        )
        self.assertEqual(result['base_frame'], 'base_link')
        self.assertFalse(result['execution_blockers'])
        np.testing.assert_allclose(np.asarray(result['T_base_object'])[:3, 3], [.5, -.2, .8])

    def test_approximate_depth_remains_blocked_after_transform(self):
        snap = self.snapshot()
        snap.notes = '头部无深度 → RGB-only（假定平面深度）'
        result = observation_from_snapshot(
            snap, 1., 'obs', base_pose=(np.zeros(3), np.eye(3), None, (.02, .03, .04))
        )
        self.assertTrue(result['execution_blockers'])

    def test_lingbot_depth_is_used_and_identified(self):
        snap = self.snapshot('base_link', True)
        snap.notes = '头部深度 ← LingBot-Depth'
        result = observation_from_snapshot(snap, 1., 'obs', depth_source='lingbot')
        self.assertFalse(result['execution_blockers'])
        self.assertEqual(result['depth_source'], 'lingbot')
        self.assertIn('LingBot 推断深度', result['source_label'])
        np.testing.assert_allclose(result['object_pose']['extents_m'], snap.obb_extents)

    def test_lingbot_does_not_bypass_other_blockers(self):
        snap = self.snapshot()
        result = observation_from_snapshot(snap, 1., 'obs', depth_source='lingbot')
        self.assertTrue(result['execution_blockers'])
        snap = self.snapshot('base_link', True)
        for note in ('RGB-only 假定平面深度', 'simgt'):
            snap.notes = note
            result = observation_from_snapshot(snap, 1., 'obs', depth_source='lingbot')
            self.assertTrue(result['execution_blockers'])


if __name__ == '__main__':
    unittest.main()
