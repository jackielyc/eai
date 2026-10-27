"""Offline Pinocchio + repository URDF test. No ROS or hardware I/O."""
import unittest

import numpy as np
import pinocchio as pin

from host_arm_ik import HostArmIK, _JOINT_NAMES, _resolve_urdf
from host_grasp_plan_worker import prepare
from grasp_execution import TCP_POSITION_TOLERANCE_M, TCP_ROTATION_TOLERANCE_RAD


class OfflineIKTests(unittest.TestCase):
    def test_shared_tcp_tolerances_are_bounded(self):
        self.assertEqual(TCP_POSITION_TOLERANCE_M, .01)
        self.assertAlmostEqual(np.degrees(TCP_ROTATION_TOLERANCE_RAD), 5.)
    def request(self, side):
        ik = HostArmIK()
        indices = [ik._q_idx[name] for name in _JOINT_NAMES]
        q = np.asarray(ik.list_from_q(pin.neutral(ik.model)))
        q = np.clip(q, ik.model.lowerPositionLimit[indices]+.01, ik.model.upperPositionLimit[indices]-.01)
        j = 3 if side == 'left' else 10
        q[j] += .2
        goal = q.copy()
        goal[j+1] += .03
        pin.forwardKinematics(ik.model, ik.data, ik.q_from_list(goal))
        pin.updateFramePlacements(ik.model, ik.data)
        fid = ik._left_fid if side == 'left' else ik._right_fid
        T = ik.data.oMf[fid].homogeneous.copy()
        return ik, {'profile': {'urdf_path': _resolve_urdf(), 'arm_side': side,
                               'workspace_m': [[-3,-3,-3], [3,3,3]],
                               'joint_velocity_rad_s': .1, 'joint_acceleration_rad_s2': .2},
                    'preview': {'id': 'offline-test', 'goals': {'approach': T.tolist()}},
                    'stage': 'approach', 'start_joints': q.tolist()}

    def test_both_arms_reachable_path_and_frozen_other_joints(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                ik, request = self.request(side)
                result = prepare(request)
                self.assertTrue(result['ik_checked'])
                self.assertFalse(result['collision_checked'])
                rows = np.asarray(result['joint_waypoints'])
                fixed = [0,1,*range(9,16)] if side == 'left' else list(range(9))
                np.testing.assert_allclose(rows[:, fixed], np.tile(rows[0, fixed], (len(rows), 1)), atol=1e-8)
                pin.forwardKinematics(ik.model, ik.data, ik.q_from_list(rows[-1]))
                pin.updateFramePlacements(ik.model, ik.data)
                fid = ik._left_fid if side == 'left' else ik._right_fid
                actual = ik.data.oMf[fid]
                target = np.asarray(request['preview']['goals']['approach'])
                self.assertLess(np.linalg.norm(actual.translation-target[:3,3]), .004)
                self.assertLess(np.linalg.norm(pin.log3(target[:3,:3].T @ actual.rotation)), .04)

    def test_far_goal_and_out_of_limit_seed_rejected(self):
        _, request = self.request('right')
        request['preview']['goals']['approach'][0][3] += 2
        with self.assertRaisesRegex(ValueError, '单段距离'): prepare(request)
        _, request = self.request('right')
        request['start_joints'][10] = 100
        with self.assertRaisesRegex(ValueError, '限位'): prepare(request)

    def test_simulation_and_hardware_share_segment_limits(self):
        _, request = self.request('right')
        request['preview']['goals']['approach'][0][3] += .61
        messages = []
        for mode in ('simulation', 'hardware'):
            request['execution_mode'] = mode
            with self.assertRaises(ValueError) as caught:
                prepare(request)
            messages.append(str(caught.exception))
            self.assertIn('上限 0.60m', messages[-1])
            self.assertIn('上限 92°', messages[-1])
        self.assertEqual(messages[0].replace('仿真', ''), messages[1].replace('真机', ''))

    def test_allow_waist_is_recorded_but_other_arm_stays_fixed(self):
        _, request = self.request('left')
        request['allow_waist'] = True
        result = prepare(request)
        self.assertTrue(result['waist_included'])
        rows = np.asarray(result['joint_waypoints'])
        np.testing.assert_allclose(rows[:, 9:16], np.tile(rows[0, 9:16], (len(rows), 1)), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
