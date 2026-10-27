import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from grasp_execution import (JointRunner, Sequence, load_profile, make_preview,
                             evaluate_wrench, manual_candidate, read_feedback,
                             sample_trajectory, timed_waypoints)


def example():
    p = json.loads(Path(__file__).with_name('config').joinpath('grasp_execution.example.json').read_text())
    p['calibrated'] = True
    p['hand_model'] = 'test_hand'
    p['workspace_m'] = [[-2, -2, -2], [2, 2, 2]]
    T = np.eye(4)
    T[:3, 3] = [.3, -.2, .8]
    obs = dict(ok=True, captured_at=time.time(), observation_id='test', base_frame='base_link',
               T_base_camera=np.eye(4).tolist(), object_pose={'T_camera_object': T.tolist()},
               grasps=dict(ok=True, gripper='test_hand', candidates=[{'T_camera_grasp': T.tolist(), 'model_score': .5}]))
    return p, obs


class GeometryTests(unittest.TestCase):
    def test_wrench_validation_detects_force_torque_stale_and_sensor_error(self):
        now = 100.
        ok = {'force':[3.,4.,0.], 'torque':[0.,0.,.2], 'force_state':0, 'stamp':now}
        violation, force, torque = evaluate_wrench(ok, 10., .5, now=now)
        self.assertIsNone(violation)
        self.assertEqual(force, 5.)
        self.assertEqual(torque, .2)
        sample = dict(ok, force=[11.,0.,0.])
        self.assertIn('合力', evaluate_wrench(sample,10.,.5,now=now)[0])
        sample = dict(ok, torque=[0.,0.,.6])
        self.assertIn('合力矩', evaluate_wrench(sample,10.,.5,now=now)[0])
        self.assertIn('过期', evaluate_wrench(dict(ok,stamp=99.),10.,.5,now=now)[0])
        self.assertIn('状态异常', evaluate_wrench(dict(ok,force_state=-1),10.,.5,now=now)[0])
    def test_transform_approach_axis_is_grasp_axis_not_tcp_axis(self):
        p, obs = example()
        p['T_grasp_tcp'] = [[0, -1, 0, .1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        p['approach_axis_grasp'] = [1, 0, 0]
        preview = make_preview(obs, 0, p)
        np.testing.assert_allclose(np.asarray(preview['goals']['approach'])[:3, 3], [.4, -.2, .8])
        np.testing.assert_allclose(np.asarray(preview['goals']['pregrasp'])[:3, 3], [.35, -.2, .8])
        self.assertEqual(preview['arm_side'], 'right')

    def test_missing_extrinsics_stale_wrong_hand_workspace_and_unknown_config(self):
        p, obs = example()
        for update, message in (({'base_frame': 'camera'}, '外参'),
                                ({'grasps': {'ok': True, 'gripper': 'other'}}, '手型')):
            value = dict(obs, **update)
            with self.assertRaisesRegex(ValueError, message): make_preview(value, 0, p)
        self.assertEqual(make_preview(dict(obs, captured_at=0), 0, p)['observation_id'], 'test')
        p['workspace_m'] = [[0, 0, 0], [.1, .1, .1]]
        with self.assertRaisesRegex(ValueError, '工作空间'): make_preview(obs, 0, p)
        with self.assertRaisesRegex(ValueError, '标定'):
            load_profile(Path(__file__).with_name('config') / 'grasp_execution.example.json')

    def test_manual_contact_compensates_tcp_offset(self):
        p, obs = example()
        p['T_grasp_tcp'][0][3] = .1
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'robot.urdf'; path.write_text('test')
            p['urdf_path'] = str(path)
            feedback = {'urdf_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'right': {'xyzw': [0, 0, 0, 1]}}
            obs['grasps'] = manual_candidate(obs, p, feedback, [0, 0, .02])
            preview = make_preview(obs, 0, p)
            np.testing.assert_allclose(np.asarray(preview['goals']['approach'])[:3, 3], [.4, -.2, .82])
            self.assertEqual(obs['grasps']['model'], 'manual_current_hand_orientation')

    def test_feedback_rejects_command_only_or_stale_sources(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'feedback.json'
            path.write_text(json.dumps({'q16': [0]*16, 't': time.time()}))
            with self.assertRaisesRegex(ValueError, 'arm_received'): read_feedback(path)
            path.write_text(json.dumps({'q16': [0]*16, 'arm_received_at': 100, 'waist_received_at': 100}))
            read_feedback(path, now=100.1)
            with self.assertRaises(ValueError): read_feedback(path, now=101)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.profile, obs = example()
        self.sequence = Sequence()
        self.sequence.reset(make_preview(obs, 0, self.profile), self.profile)

    def accepted(self, stage):
        return {'ok': True, 'preview_id': self.sequence.preview['id'], 'stage': stage,
                'backend': 'host_ik_manual', 'ik_checked': True, 'joint_limits_checked': True,
                'collision_checked': False}

    def test_full_order_never_automatically_closes_or_lifts(self):
        s = self.sequence
        with self.assertRaises(ValueError): s.begin_plan()
        s.hand_sent()
        with self.assertRaises(ValueError): s.begin_plan()
        s.confirm_hand()
        for stage in ('pregrasp', 'approach'):
            self.assertEqual(s.begin_plan(), stage)
            s.accept_plan(self.accepted(stage), now=10)
            with self.assertRaisesRegex(ValueError, '碰撞'): s.begin_execute(now=11)
            s.begin_execute(now=11, manual_path_confirmed=True)
            s.motion_finished(True)
        self.assertEqual(s.state, 'at_grasp')
        s.hand_sent()
        with self.assertRaises(ValueError): s.begin_plan()
        s.confirm_hand()
        self.assertEqual(s.begin_plan(), 'lift')
        s.accept_plan(self.accepted('lift'), now=20)
        s.begin_execute(now=21, manual_path_confirmed=True)
        s.motion_finished(True)
        self.assertEqual(s.state, 'complete')

    def test_stopped_late_result_and_expired_plan_cannot_run(self):
        s = self.sequence
        s.hand_sent(); s.confirm_hand(); s.begin_plan()
        s.stop()
        with self.assertRaises(ValueError): s.accept_plan(self.accepted('pregrasp'))
        s.reset(s.preview, self.profile)
        s.hand_sent(); s.confirm_hand(); s.begin_plan()
        s.accept_plan(self.accepted('pregrasp'), now=0)
        with self.assertRaisesRegex(ValueError, '过期'): s.begin_execute(now=100 + self.profile['max_plan_age_s'], manual_path_confirmed=True)
        self.assertEqual(s.state, 'failed')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.sent = []
        q0, q1 = np.zeros(16), np.zeros(16)
        q1[9] = .02
        T = np.eye(4)
        self.plan = {'joint_waypoints': [q0.tolist(), q1.tolist()], 'times_s': [0, 1],
                     'urdf_sha256': 'model', 'target': T.tolist(), 'arm_side': 'right'}
        self.data = dict(q16=q0.tolist(), urdf_sha256='model', arm_received_at=0,
                         right={'xyz': [0, 0, 0], 'xyzw': [0, 0, 0, 1]})
        self.runner = JointRunner(self.plan, self.sent.append, lambda: self.data, lambda: True, clock=lambda: self.now)

    def test_end_of_command_stream_is_not_success(self):
        self.now = 1.
        self.runner.tick()
        self.assertEqual(self.runner.state, 'running')
        for i in range(3):
            self.data['q16'] = self.plan['joint_waypoints'][-1]
            self.data['arm_received_at'] = i+1
            self.now += .02
            self.runner.tick()
        self.assertEqual(self.runner.state, 'complete')

    def test_cancel_holds_and_waits_for_three_distinct_feedback_samples(self):
        self.runner.stop()
        for _ in range(4): self.runner.tick()
        self.assertEqual(self.runner.state, 'stopping')
        for i in range(3):
            self.data['arm_received_at'] = i+1
            self.now += .02
            self.runner.tick()
        self.assertEqual(self.runner.state, 'stopped')
        np.testing.assert_array_equal(self.sent[-1], [0]*16)

    def test_pause_resume_freezes_trajectory_clock(self):
        self.now = .3
        self.runner.tick()
        held = self.sent[-1].copy()
        self.data['q16'] = held
        self.runner.pause()
        for _ in range(4):
            self.runner.tick()
        self.assertEqual(self.runner.state, 'pausing')
        for i in range(3):
            self.data['arm_received_at'] = i+1
            self.now += .02
            self.runner.tick()
        self.assertEqual(self.runner.state, 'paused')
        self.now = 100
        self.runner.tick()
        np.testing.assert_allclose(self.sent[-1], held)
        self.runner.resume()
        self.runner.tick()
        np.testing.assert_allclose(self.sent[-1], held)
        self.assertEqual(self.runner.state, 'running')

    def test_pause_can_be_cancelled_and_drift_prevents_resume(self):
        self.runner.pause()
        for i in range(3):
            self.data['arm_received_at'] = i+1
            self.runner.tick()
        self.data['q16'][9] = .03
        with self.assertRaisesRegex(ValueError, '位置发生变化'):
            self.runner.resume()
        self.runner.stop()
        self.assertEqual(self.runner.state, 'stopping')

    def test_waist_may_move_only_when_plan_explicitly_includes_it(self):
        plan = dict(self.plan, waist_included=True,
                    joint_waypoints=[[0.]*16, [.02, .02]+[0.]*14])
        runner = JointRunner(plan, self.sent.append, lambda:self.data, lambda:True,
                             clock=lambda:self.now)
        self.now = .5
        runner.tick()
        self.assertEqual(runner.state, 'running')
        self.assertGreater(self.sent[-1][0], 0.)
        plan['waist_included'] = False
        runner = JointRunner(plan, self.sent.append, lambda:self.data, lambda:True,
                             clock=lambda:self.now)
        self.data['q16'][0] = .04
        runner.tick()
        self.assertEqual(runner.state, 'stopping')

    def test_start_drift_and_model_mismatch_rejected(self):
        self.data['q16'][0] = .1
        with self.assertRaisesRegex(ValueError, '起点'):
            JointRunner(self.plan, self.sent.append, lambda: self.data, lambda: True)
        self.data['urdf_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, '模型'):
            JointRunner(self.plan, self.sent.append, lambda: self.data, lambda: True)

    def test_timing_respects_velocity_and_acceleration_bounds(self):
        rows = np.array([[0]*16, [.08]*16, [.02]*16])
        times = timed_waypoints(rows, .1, .2)
        dt = .001
        values = np.array([sample_trajectory(rows, times, t)[0] for t in np.arange(0, times[-1], dt)])
        self.assertLessEqual(np.abs(np.diff(values)/dt).max(), .1001)
        self.assertLessEqual(np.abs(np.diff(values, n=2)/dt**2).max(), .2001)

    def test_stale_feedback_does_not_fake_a_stopped_confirmation(self):
        def stale(): raise ValueError('stale')
        self.runner.feedback = stale
        self.runner.tick()
        self.now = 4
        self.runner.tick()
        self.assertEqual(self.runner.state, 'stop_unconfirmed')
        self.assertFalse(self.sent)


if __name__ == '__main__':
    unittest.main()
