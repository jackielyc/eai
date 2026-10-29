"""Bounded full-pose and dual-TCP regressions, offline only."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import least_squares
from host_arm_ik import HostArmIK, _JOINT_NAMES, _resolve_urdf
from host_grasp_plan_worker import prepare
from grasp_execution import IK_ADJACENT_JUMP_LIMIT_RAD, sample_trajectory
import test_host_grasp_plan as offline_cases


class BoundedIKTests(unittest.TestCase):
    def setUp(self):
        self.ik = HostArmIK()
        self.pin = self.ik._pin

    def test_weighted_pose_jacobian_matches_finite_difference(self):
        q = np.zeros(16); q[0] = .2; q[1] = .1
        goal_q=q.copy(); goal_q[3]=.3; goal_q[5]=.2
        self.ik.ee_pose(goal_q, 'left')
        goal=self.ik.data.oMf[self.ik._left_fid].copy()
        checked=[]

        def verify(fun, x0, **kwargs):
            analytical=kwargs['jac'](x0).copy()
            numerical=np.zeros_like(analytical)
            eps=1e-6
            for i in range(len(x0)):
                a=x0.copy(); b=x0.copy(); a[i]+=eps; b[i]-=eps
                numerical[:,i]=(fun(a).copy()-fun(b).copy())/(2*eps)
            np.testing.assert_allclose(analytical,numerical,atol=1e-4,rtol=1e-4)
            checked.append(True)
            return least_squares(fun,x0,**kwargs)

        with patch('scipy.optimize.least_squares',side_effect=verify):
            result=self.ik.solve_targets(q,[(self.ik._left_fid,goal)],_JOINT_NAMES[2:9])
        self.assertTrue(checked)
        self.assertIsNotNone(result)

    def test_small_move_still_includes_waist_and_both_tcp_tasks(self):
        q=np.zeros(16); q[0]=.2
        self.ik.ee_pose(q,'left')
        target=self.ik.data.oMf[self.ik._left_fid].copy()
        target.translation[0]+=.001
        with patch.object(self.ik,'solve_targets',return_value=q.tolist()) as solve:
            self.ik.solve_coordinated(q,moving_side='left',target=target,allow_waist=True)
        self.assertEqual(solve.call_args.args[2],_JOINT_NAMES)
        self.assertEqual(len(solve.call_args.args[1]),2)

    def test_no_position_only_success_for_incompatible_orientation(self):
        q=np.zeros(16); q[0]=.2
        self.ik.ee_pose(q,'left')
        target=self.ik.data.oMf[self.ik._left_fid].copy()
        other=target.copy(); other.rotation=target.rotation@self.pin.exp3(np.array([.5,0,0]))
        result=self.ik.solve_targets(q,[(self.ik._left_fid,target),(self.ik._left_fid,other)],
                                     _JOINT_NAMES[2:9],max_iters=50)
        self.assertIsNone(result)

    def test_path_subdivision_retries_without_relaxing_limits(self):
        _, request = offline_cases.OfflineIKTests().request('left')
        original = HostArmIK.solve_coordinated
        attempts = []
        def fail_once(ik, *args, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                return None
            return original(ik, *args, **kwargs)
        with patch.object(HostArmIK, 'solve_coordinated', fail_once):
            result = prepare(request)
        self.assertGreater(result['path_refinements'], 0)
        self.assertTrue(all(a['pos_tol'] == .002 and a['ori_tol'] == .02 for a in attempts))

    def test_pregrasp_can_fall_back_to_joint_guided_coupled_path(self):
        _, request = offline_cases.OfflineIKTests().request('left')
        request['allow_waist'] = True
        request['preview']['goals']['pregrasp'] = request['preview']['goals'].pop('approach')
        request['stage'] = 'pregrasp'
        original = HostArmIK.solve_coordinated
        calls = []
        def fail_first_branch(ik, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) <= 6:
                return None
            return original(ik, *args, **kwargs)
        with patch.object(HostArmIK, 'solve_coordinated', fail_first_branch):
            result = prepare(request)
        self.assertEqual(result['path_kind'], 'joint_guided_pregrasp')
        self.assertTrue(result['other_tcp_held'])

    def test_failed_field_requests_now_satisfy_full_path_constraints(self):
        cases=json.loads(Path(__file__).with_name('test_data').joinpath('ik_regressions.json').read_text())
        indices=[self.ik._q_idx[n] for n in _JOINT_NAMES]
        lo,hi=self.ik.model.lowerPositionLimit[indices],self.ik.model.upperPositionLimit[indices]
        for case in cases:
            with self.subTest(record=case['id']):
                profile=dict(urdf_path=_resolve_urdf(),arm_side=case['side'],workspace_m=[[0,-.5,.4],[1.5,.5,1.3]],
                             joint_velocity_rad_s=.1,joint_acceleration_rad_s2=.2)
                result=prepare(dict(profile=profile,preview={'id':case['id'],'goals':{case['stage']:case['target']}},
                                    stage=case['stage'],start_joints=case['start_joints'],allow_waist=True))
                self.assertEqual(result['solver'],'bounded_dual_tcp')
                self.assertFalse(result['collision_checked'])
                rows=np.asarray(result['joint_waypoints']);times=np.asarray(result['times_s'])
                self.assertTrue(np.all(rows>=lo-1e-9) and np.all(rows<=hi+1e-9))
                self.assertLessEqual(np.abs(np.diff(rows,axis=0)).max(),IK_ADJACENT_JUMP_LIMIT_RAD+1e-8)
                np.testing.assert_allclose(rows[0],case['start_joints'])
                held=np.asarray(result['held_target'])
                held_fid=self.ik._right_fid if case['side']=='left' else self.ik._left_fid
                active_fid=self.ik._left_fid if case['side']=='left' else self.ik._right_fid
                for t in np.linspace(0,times[-1],150):
                    q=sample_trajectory(rows,times,t)
                    self.pin.forwardKinematics(self.ik.model,self.ik.data,self.ik.q_from_list(q))
                    self.pin.updateFramePlacements(self.ik.model,self.ik.data)
                    actual=self.ik.data.oMf[held_fid]
                    self.assertLess(np.linalg.norm(actual.translation-held[:3,3]),.015)
                    self.assertLess(np.linalg.norm(self.pin.log3(held[:3,:3].T@actual.rotation)),np.deg2rad(10))
                final=self.ik.data.oMf[active_fid];target=np.asarray(case['target'])
                self.assertLess(np.linalg.norm(final.translation-target[:3,3]),.0021)
                self.assertLess(np.linalg.norm(self.pin.log3(target[:3,:3].T@final.rotation)),.0201)


if __name__=='__main__':
    unittest.main()
