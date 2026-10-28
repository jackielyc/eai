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
from grasp_task_workflow import TASKS
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
        self.wrench_samples = {
            side: {'force':[0.,0.,0.], 'torque':[0.,0.,0.],
                   'force_state':0}
            for side in ('left','right')
        }
        def wrench_feedback(side):
            sample = self.wrench_samples.get(side)
            if sample is None:
                return None
            result = dict(sample)
            result.setdefault('stamp', __import__('time').time())
            return result
        self.panel = GraspExecutionPanel(
            self.node, busy=lambda: False,
            wrench_feedback=wrench_feedback,
            ensure_wrench_monitor=lambda:None,
        )
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

    def test_task_creation_controls_are_at_top_and_disable_while_busy(self):
        self.assertEqual(self.panel.task_new_button.text(), '新建任务…')
        self.assertEqual(self.panel.task_copy_button.text(), '复制任务…')
        self.assertEqual(self.panel.task_manage_button.text(), '管理任务…')
        self.panel.process = SimpleNamespace()
        self.panel.refresh()
        self.assertFalse(self.panel.task_new_button.isEnabled())
        self.assertFalse(self.panel.task_copy_button.isEnabled())
        self.assertFalse(self.panel.task_manage_button.isEnabled())
        self.panel.process = None

    def test_all_execute_and_pause_buttons_toggle_state(self):
        self.assertEqual(self.panel.workflow_all_button.text(),'全部执行')
        self.assertEqual(self.panel.workflow_auto_pause_button.text(),'暂停')
        self.panel.set_target_observations({
            'target-1':{'label':'手机','methods':{'pca':self.obs}},
            'target-2':{'label':'盒子','methods':{'pca':dict(self.obs,observation_id='box')}},
            'target-3':{'label':'盒盖','methods':{'pca':dict(self.obs,observation_id='lid')}},
        })
        self.panel.task_role_combos['phone'].setCurrentIndex(1)
        self.panel.task_role_combos['box'].setCurrentIndex(2)
        self.panel.task_role_combos['lid'].setCurrentIndex(3)
        self.panel.execution_mode.setCurrentIndex(0)
        with patch.object(self.panel,'_capture_auto_workflow_start',return_value={'q16':[0]*16}), \
             patch.object(self.panel,'_schedule_auto_workflow') as schedule:
            self.panel.workflow_all_button.click()
            schedule.assert_called_once_with(0)
        self.assertTrue(self.panel._workflow_auto_running)
        self.panel.refresh();self.assertEqual(self.panel.workflow_all_button.text(),'取消执行')
        self.panel.toggle_auto_workflow_pause()
        self.assertTrue(self.panel._workflow_auto_paused)
        self.panel.refresh();self.assertEqual(self.panel.workflow_auto_pause_button.text(),'继续执行')

    def test_cancel_all_restores_captured_simulation_state_and_resets_flow(self):
        from robot_view3d import _resolve_urdf
        self.panel.execution_mode.setCurrentIndex(0)
        self.profile['urdf_path']=_resolve_urdf()
        Path(self.panel.profile_path.text()).write_text(json.dumps(self.profile))
        self.panel.simulation_display=lambda q,h:None
        self.panel.simulation=None
        sim=self.panel.simulation_backend()
        initial=dict(sim.feedback(),hands={'left':[.1]*6,'right':[.2]*6})
        sim.q[2]=.03;sim.q[9]=-.03;sim.hands={'left':[.8]*6,'right':[.7]*6}
        self.panel._workflow_auto_initial_snapshot=initial
        self.panel._workflow_auto_running=True
        self.panel._workflow_step=6;self.panel.task_flow.set_current(6)
        self.panel.cancel_all_workflow()
        self.assertIsNotNone(self.panel.runner,self.panel.message)
        limit=int(self.panel.runner.times[-1]/.02)+20
        for _ in range(limit):
            if self.panel.runner is None: break
            self.panel.runner.started-=.02
            self.panel.monitor()
        self.assertIsNone(self.panel.runner)
        self.assertFalse(self.panel._workflow_auto_running)
        self.assertEqual(self.panel._workflow_step,0)
        np.testing.assert_allclose(sim.q[2:],initial['q16'][2:])
        self.assertEqual(sim.hands,initial['hands'])
        self.assertIn('已取消并恢复',self.panel.message)

    def test_all_execute_chains_immediate_workflow_steps_to_completion(self):
        from robot_view3d import _resolve_urdf
        key='task_auto_immediate'
        TASKS[key]={'name':'自动链测试','builtin':False,'roles':(('item','物品'),),
                    'steps':({'label':'选择','kind':'target','role':'item'},
                             {'label':'张开','kind':'hand','role':'item','open':True},
                             {'label':'确认','kind':'confirm'},
                             {'label':'完成','kind':'complete'})}
        try:
            self.panel.refresh_task_selector(key)
            self.panel.set_target_observations({'target-1':{'label':'物品',
                'methods':{'pca':self.obs}}})
            self.panel.task_role_combos['item'].setCurrentIndex(1)
            self.panel.execution_mode.setCurrentIndex(0)
            self.panel.simulation_display=lambda q,h:None
            self.profile['urdf_path']=_resolve_urdf()
            Path(self.panel.profile_path.text()).write_text(json.dumps(self.profile))
            self.panel.simulation=None
            with patch.object(self.panel,'_schedule_auto_workflow'):
                self.panel.toggle_all_workflow()
                for _ in range(4): self.panel._run_auto_workflow_next()
            self.assertFalse(self.panel._workflow_auto_running)
            self.assertEqual(self.panel._workflow_step,3)
            self.assertIn('全部流程执行完成',self.panel.message)
        finally:
            TASKS.pop(key,None)

    def test_copy_task_immediately_creates_and_selects_named_copy(self):
        key='task_test_copy'
        try:
            with patch('grasp_execution_ui.QInputDialog.getText', return_value=('装配流程副本',True)), \
                 patch('grasp_execution_ui.new_task_key', return_value=key), \
                 patch('grasp_execution_ui.save_tasks') as save:
                self.panel.copy_task()
            save.assert_called_once()
            self.assertIn(key,TASKS)
            self.assertEqual(TASKS[key]['name'],'装配流程副本')
            self.assertFalse(TASKS[key]['builtin'])
            self.assertEqual(self.panel.task_combo.currentData(),key)
            self.assertIn('复制成功',self.panel.task_data_status.text())
        finally:
            TASKS.pop(key,None)

    def test_multiple_targets_can_be_assigned_to_phone_box_workflow(self):
        phone = copy.deepcopy(self.obs); phone['observation_id'] = 'phone-observation'
        box = copy.deepcopy(self.obs); box['observation_id'] = 'box-observation'
        self.panel.set_target_observations({
            'target-1': {'label':'目标 1', 'methods':{'pca':phone}},
            'target-2': {'label':'目标 2', 'methods':{'pca':box}},
            'target-3': {'label':'目标 3', 'methods':{'pca':dict(self.obs,observation_id='lid')}},
        })
        self.assertEqual(self.panel.grasp_target_combo.count(), 3)
        self.panel.task_role_combos['phone'].setCurrentIndex(1)
        self.panel.task_role_combos['box'].setCurrentIndex(2)
        self.panel.task_role_combos['lid'].setCurrentIndex(3)
        self.panel.workflow_next_button.click()
        self.assertEqual(self.panel._workflow_step, 0)
        self.assertEqual(self.panel._workflow_substep, 1)
        self.assertEqual(self.panel.observation['observation_id'], 'phone-observation')
        with patch.object(self.panel, 'execute', side_effect=lambda:setattr(self.panel,'process',SimpleNamespace())) as execute:
            self.panel.workflow_next_button.click()
            execute.assert_called_once()
        self.assertTrue(self.panel._workflow_waiting)
        self.assertEqual(self.panel.motion_stage.currentData(), 'pregrasp')
        self.assertEqual(self.panel.grasp_target_combo.currentData(), 'target-1')
        self.panel.process = None

    def test_phone_and_box_roles_must_be_distinct(self):
        self.panel.set_target_observations({
            'target-1': {'label':'目标 1', 'methods':{'pca':self.obs}},
        })
        self.panel.task_role_combos['phone'].setCurrentIndex(1)
        self.panel.task_role_combos['box'].setCurrentIndex(1)
        self.panel._workflow_step = 1
        self.panel._workflow_substep = 0
        self.panel.task_flow.set_current(1)
        self.panel.execute_workflow_next()
        self.assertIn('不能相同', self.panel.message)
        self.assertFalse(self.panel._workflow_waiting)

    def test_flowchart_node_click_selects_step_and_is_blocked_while_running(self):
        self.panel._workflow_step=0;self.panel._workflow_substep=3
        self.panel.task_flow._labels[2].clicked.emit(2)
        self.assertEqual(self.panel._workflow_step,2)
        self.assertEqual(self.panel._workflow_substep,0)
        self.assertIn('盖上手机盒',self.panel.message)
        self.assertIn('选择盒盖',self.panel.message)
        self.panel.process=SimpleNamespace()
        self.panel.task_flow._labels[0].clicked.emit(0)
        self.assertEqual(self.panel._workflow_step,2)
        self.assertIn('不能跳转',self.panel.message)
        self.panel.process=None

    def test_task_save_and_load_bind_targets_roles_and_workflow_step(self):
        saved=[]
        self.panel.save_task_data=lambda key,state:saved.append((key,state)) or '保存成功'
        self.panel.set_target_observations({
            'target-1':{'label':'手机','methods':{'pca':self.obs}},
            'target-2':{'label':'盒子','methods':{'pca':dict(self.obs,observation_id='box')}},
            'target-3':{'label':'盒盖','methods':{'pca':dict(self.obs,observation_id='lid')}},
        })
        self.panel.task_role_combos['phone'].setCurrentIndex(1)
        self.panel.task_role_combos['box'].setCurrentIndex(2)
        self.panel.task_role_combos['lid'].setCurrentIndex(3)
        self.panel._workflow_step=2;self.panel._workflow_substep=4;self.panel.task_flow.set_current(2)
        self.panel.grasp_arm.setCurrentIndex(0)
        self.panel.waist_participation.setChecked(True)
        self.panel.force_limit.setValue(22.)
        self.panel.task_save_button.click()
        self.assertEqual(saved[0][0],'phone_box_assembly')
        self.assertEqual(saved[0][1]['roles'],{'phone':'target-1','box':'target-2','lid':'target-3'})
        self.assertEqual(saved[0][1]['workflow_step'],2)
        self.assertEqual(saved[0][1]['workflow_substep'],4)
        self.assertEqual(saved[0][1]['step_settings']['2:4']['arm_side'],'left')
        self.assertTrue(saved[0][1]['step_settings']['2:4']['allow_waist'])
        self.assertEqual(saved[0][1]['step_settings']['2:4']['force_limit_n'],22.)
        self.assertIn('保存成功',self.panel.task_data_status.text())

        self.panel.load_task_data=lambda key:{'saved_at_text':'2026-10-27 18:00:00',
            'target_observations':self.panel.target_observations,
            'panel_state':{'roles':{'phone':'target-1','box':'target-2','lid':'target-3'},
                           'selected_target':'target-2','workflow_step':2,'workflow_substep':6,
                           'arm_side':'left','allow_waist':True,'motion_stage':'approach',
                           'step_settings':{'2:6':{'arm_side':'right','allow_waist':True,
                               'pose_method':'pca','execution_mode':'simulation','motion_stage':'approach',
                               'target_id':'target-2','contact_offsets_cm':[1.,2.,3.],
                               'wrench_guard':True,'force_limit_n':25.,'torque_limit_nm':1.5}}}}
        self.panel.change_task_workflow()
        self.assertEqual(self.panel._workflow_step,2)
        self.assertEqual(self.panel._workflow_substep,6)
        self.assertEqual(self.panel.grasp_target_combo.currentData(),'target-2')
        self.assertEqual(self.panel.grasp_arm.currentData(),'right')
        self.assertTrue(self.panel.waist_participation.isChecked())
        self.assertEqual([spin.value() for spin in self.panel.contact_offsets],[1.,2.,3.])
        self.assertEqual(self.panel.force_limit.value(),25.)
        self.assertIn('已加载',self.panel.task_data_status.text())

    def test_wrench_limit_pauses_and_requires_manual_safe_resume(self):
        class Runner:
            state='running'; message=''; plan={'arm_side':'right'}
            def tick(self): pass
            def pause(self): self.state='paused'; self.message='paused'
            def resume(self): self.state='running'; self.message='running'
        self.panel.runner = Runner()
        self.panel.sequence.state = 'executing'
        self.wrench_samples['right'] = {
            'force':[16.,0.,0.], 'torque':[0.,0.,0.],
            'force_state':0, 'stamp':__import__('time').time()}
        self.panel.monitor()
        self.assertEqual(self.panel.runner.state, 'paused')
        self.assertIn('六维力保护触发', self.panel.message)
        self.panel.pause_or_resume()
        self.assertEqual(self.panel.runner.state, 'paused')
        self.assertIn('仍不满足', self.panel.message)
        self.wrench_samples['right'] = {
            'force':[2.,0.,0.], 'torque':[0.,0.,.1],
            'force_state':0, 'stamp':__import__('time').time()}
        self.panel.pause_or_resume()
        self.assertEqual(self.panel.runner.state, 'running')
        self.assertIsNone(self.panel._wrench_guard_triggered)

    def test_real_execution_requires_fresh_wrench_when_guard_enabled(self):
        self.wrench_samples['right'] = None
        self.panel.sequence.profile = self.profile
        self.panel.sequence.preview = dict(id='p', arm_side='right', hand_model='test_hand',
                                           tcp_link='Link7_r', base_frame='base_link',
                                           goals={'approach':np.eye(4).tolist()})
        self.panel.sequence.pending_stage = 'pregrasp'
        self.panel.sequence.state = 'planned'
        self.panel.sequence.plan = dict(
            accepted_at=__import__('time').monotonic(), arm_side='right', stage='pregrasp')
        self.panel.scene_confirm.setChecked(True)
        self.panel._execute_ready_plan()
        self.assertIsNone(self.panel.runner)
        self.assertIn('六维力保护未就绪', self.panel.message)
        self.assertFalse(self.node._grasp_execution_reserved)

    def test_selected_grasp_arm_controls_profile_preview_and_hand(self):
        self.assertEqual(self.panel.grasp_arm.currentData(), 'right')
        self.panel.grasp_arm.setCurrentIndex(0)
        profile = self.panel.selected_profile()
        self.assertEqual(profile['arm_side'], 'left')
        self.assertEqual(profile['tcp_link'], 'Link7_l')
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.assertEqual(self.panel.sequence.preview['arm_side'], 'left')
        self.assertEqual(self.panel.sequence.preview['tcp_link'], 'Link7_l')
        self.panel.scene_confirm.setChecked(True)
        self.panel.hand_button.click()
        self.assertEqual(self.node.hand_commands, [('left', self.profile['open_positions'])])

    def test_switching_arm_invalidates_current_orientation_candidate(self):
        observation = copy.deepcopy(self.obs)
        observation['grasps']['model'] = 'manual_current_hand_orientation'
        observation['grasps']['candidates'][0]['source'] = 'manual'
        self.panel.set_observation(observation)
        self.assertEqual(self.panel.candidates.count(), 1)
        self.panel.grasp_arm.setCurrentIndex(0)
        self.assertNotIn('grasps', self.panel.observation)
        self.assertEqual(self.panel.candidates.count(), 0)
        self.assertIsNone(self.panel.sequence.preview)

    def test_restore_simulation_returns_both_arms_and_hands_without_hardware(self):
        from robot_view3d import _resolve_urdf
        frames = []
        self.panel.simulation_display = lambda q,h: frames.append(copy.deepcopy((q,h)))
        self.panel.execution_mode.setCurrentIndex(0)
        self.profile['urdf_path'] = _resolve_urdf()
        Path(self.panel.profile_path.text()).write_text(json.dumps(self.profile))
        self.panel._startup_seed = [0.] * 16
        self.panel._startup_snapshot = None
        self.panel.simulation = None
        sim = self.panel.simulation_backend()
        sim.q[0] = .1  # Restoration must leave waist where it currently is.
        sim.q[2], sim.q[9] = .02, -.02
        sim.hands = {'left':[.4]*6, 'right':[.3]*6}
        self.panel.scene_confirm.setChecked(True)
        self.panel.restore_button.click()
        self.assertIsNotNone(self.panel.runner, self.panel.message)
        self.assertEqual(self.panel.runner.plan['arm_side'],'both')
        self.panel.runner.started -= self.panel.runner.times[-1]
        for _ in range(4): self.panel.monitor()
        self.assertIsNone(self.panel.runner)
        np.testing.assert_allclose(sim.q[2:], [0.]*14)
        self.assertEqual(sim.q[0], .1)
        self.assertEqual(sim.hands, {'left':[0.]*6,'right':[0.]*6})
        self.assertIn('已恢复', self.panel.message)
        self.assertFalse(self.node.hand_commands)
        self.assertFalse(self.node.arm_commands)

    def test_real_restore_rejects_missing_startup_feedback(self):
        self.panel._startup_snapshot = None
        self.panel.restore_button.click()
        self.assertIn('启动时未采集', self.panel.message)
        self.assertIsNone(self.panel.runner)
        self.assertFalse(self.node.hand_commands)

    def test_cancelling_restore_does_not_restore_hands_after_stop(self):
        self.panel.execution_mode.setCurrentIndex(0)
        self.panel._restoring = True
        self.panel._restore_hands_goal = {'left':[.2]*6,'right':[.3]*6}
        runner = SimpleNamespace(state='running', message='cancelled', tick=lambda:None)
        runner.stop = lambda: setattr(runner, 'state', 'stopped')
        self.panel.runner = runner
        self.panel.sequence.state = 'executing'
        self.panel.sequence.pending_stage = 'restore'
        with patch.object(self.panel, 'finish_restore_hands') as finish:
            self.panel.stop()
            self.panel.monitor()
            finish.assert_not_called()
        self.assertFalse(self.panel._restoring)
        self.assertIsNone(self.panel._restore_hands_goal)

    def test_startup_snapshot_is_not_overwritten(self):
        import time
        self.panel._startup_recording = True
        self.panel._startup_deadline = time.monotonic()+10
        self.panel._startup_snapshot = None
        with patch('grasp_execution_ui.read_feedback', return_value={'q16':[.1]*16,'urdf_sha256':'test'}), \
             patch.object(self.panel, 'read_startup_hands', return_value={'left':[.2]*6,'right':[.3]*6}):
            self.panel.capture_startup_state()
        with patch('grasp_execution_ui.read_feedback', return_value={'q16':[.9]*16}):
            self.panel.capture_startup_state()
        self.assertEqual(self.panel._startup_snapshot['q16'], [.1]*16)

    def test_startup_arms_are_kept_when_hand_feedback_is_missing(self):
        import time
        self.panel._startup_recording = True
        self.panel._startup_deadline = time.monotonic()-1
        self.panel._startup_snapshot = None
        measured={'q16':[.2]*16,'urdf_sha256':'model'}
        with patch('grasp_execution_ui.read_feedback',return_value=measured), \
             patch.object(self.panel,'read_startup_hands',side_effect=ValueError('missing')):
            self.panel.capture_startup_state()
        self.assertEqual(self.panel._startup_snapshot['q16'],[.2]*16)
        self.assertEqual(self.panel._startup_snapshot['hands'],{})
        self.assertFalse(self.panel._startup_recording)
        self.panel._restore_hands_goal={}
        self.assertIn('双手恢复已跳过',self.panel.finish_restore_hands())

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

    def test_simulation_execute_does_not_require_hardware_path_checkbox(self):
        self.panel.execution_mode.setCurrentIndex(0)
        self.panel.scene_confirm.setChecked(False)
        self.panel.sequence.profile = self.profile
        self.panel.sequence.preview = dict(id='preview', arm_side='right', hand_model='test_hand',
                                           tcp_link='Link7_r', base_frame='base_link',
                                           goals={'approach':np.eye(4).tolist()})
        self.panel.sequence.state = 'planned'
        self.panel.sequence.pending_stage = 'pregrasp'
        self.panel.sequence.plan = dict(
            ok=True, preview_id='preview', stage='pregrasp', accepted_at=__import__('time').monotonic(),
            joint_waypoints=[[0.]*16,[0.]*16], times_s=[0,1],
            urdf_sha256='model', arm_side='right', target=np.eye(4).tolist())
        with patch('grasp_execution_ui.JointRunner') as runner:
            self.panel.simulation = SimpleNamespace(
                publish=lambda q:None, q=[0.]*16,
                feedback=lambda:{'q16':[0.]*16, 'urdf_sha256':'model'})
            self.panel._execute_ready_plan()
            runner.assert_called_once()
        self.assertEqual(self.panel.sequence.state, 'executing')

    def test_successful_preflight_keeps_confirmation_and_allows_execute_click(self):
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.panel.scene_confirm.setChecked(True)
        seq = self.panel.sequence
        seq.hand_sent()
        seq.confirm_hand()
        seq.begin_plan()
        result = dict(ok=True, preview_id=seq.preview['id'], stage='pregrasp',
                      arm_side='right',
                      backend='host_ik_manual', joint_limits_checked=True,
                      ik_checked=True, times_s=[0, 1])
        self.panel.job_directory = Path(self.tmp.name)
        (self.panel.job_directory / 'result.json').write_text(json.dumps(result))
        self.panel.process = SimpleNamespace(deleteLater=lambda: None)
        with patch.object(self.panel, 'read_output'):
            self.panel.finished(0, 0)
        self.assertTrue(self.panel.scene_confirm.isChecked())
        self.assertTrue(self.panel.execute_button.isEnabled())
        self.assertEqual(self.panel.execution_feedback.text(), self.panel.message)
        self.assertIn('IK/限位/连续性预检通过', self.panel.execution_feedback.text())
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
        self.assertTrue(self.panel.execute_button.isEnabled())  # Execution will replan.

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
        self.assertTrue(self.panel.execute_button.isEnabled())

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
        self.assertTrue(self.panel.hand_button.isEnabled())  # Hand verification is independent of arm geometry.
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

    def test_open_is_explicit_but_confirmation_does_not_gate_ik(self):
        self.panel.set_observation(self.obs)
        self.panel.preview()
        self.panel.hand_button.click()
        self.assertFalse(self.node.hand_commands)
        self.panel.scene_confirm.setChecked(True)
        self.panel.hand_button.click()
        self.assertEqual(self.node.hand_commands, [('right', self.profile['open_positions'])])
        self.assertEqual(self.panel.sequence.state, 'await_open')
        self.assertTrue(self.panel.plan_button.isEnabled())
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
        self.panel.set_reserved(True)  # A locked planning/execution target cannot be replaced.
        other = dict(self.obs, observation_id='other')
        self.panel.set_observation(other)
        self.assertEqual(self.panel.sequence.preview['observation_id'], 'test')
        self.panel.stop()

    def test_hand_and_confirmation_work_without_pose_or_preview(self):
        self.panel.scene_confirm.setChecked(True)
        self.panel.hand_button.click()
        self.assertEqual(self.node.hand_commands, [('right', self.profile['open_positions'])])
        self.assertIsNone(self.panel.sequence.preview)
        self.panel.confirm_button.click()
        self.assertIn('已记录', self.panel.message)
        self.assertFalse(self.node.arm_commands)

    def test_direct_execute_automatically_prechecks_selected_stage(self):
        self.panel.set_observation(self.obs)
        self.panel.motion_stage.setCurrentIndex(1)
        self.panel.waist_participation.setChecked(True)
        self.panel.scene_confirm.setChecked(True)
        with patch.object(self.panel, 'launch') as launch:
            self.panel.execute_button.click()
        self.assertEqual(self.panel.sequence.state, 'planning')
        self.assertTrue(self.panel._execute_after_plan)
        self.assertEqual(launch.call_args.kwargs['stage'], 'approach')
        self.assertTrue(launch.call_args.kwargs['allow_waist'])
        self.assertFalse(self.node.hand_commands)
        self.assertFalse(self.node.arm_commands)
        self.panel.job_directory = Path(self.tmp.name)
        result = dict(ok=True, preview_id=self.panel.sequence.preview['id'], stage='approach',
                      backend='host_ik_manual', ik_checked=True, joint_limits_checked=True, times_s=[0,1])
        (self.panel.job_directory/'result.json').write_text(json.dumps(result))
        self.panel.process = SimpleNamespace(deleteLater=lambda:None)
        with patch.object(self.panel, 'read_output'), patch.object(self.panel, '_execute_ready_plan') as execute:
            self.panel.finished(0,0)
            execute.assert_called_once()

    def test_standalone_precheck_does_not_execute_and_cancel_revokes_execution(self):
        self.panel.set_observation(self.obs)
        self.panel.motion_stage.setCurrentIndex(2)
        with patch.object(self.panel, 'launch') as launch:
            self.panel.plan_button.click()
        self.assertEqual(launch.call_args.kwargs['stage'], 'lift')
        self.assertFalse(self.panel._execute_after_plan)

    def test_waist_toggle_invalidates_plan_and_is_locked_while_busy(self):
        self.panel.sequence.state = 'planned'
        self.panel.sequence.plan = {'accepted_at': 1}
        self.panel.scene_confirm.setChecked(True)
        self.panel.waist_participation.setChecked(True)
        self.assertEqual(self.panel.sequence.state, 'idle')
        self.assertIsNone(self.panel.sequence.plan)
        self.assertFalse(self.panel.scene_confirm.isChecked())
        self.assertIn('腰部参与', self.panel.message)
        self.panel.process = SimpleNamespace()
        self.panel.refresh()
        self.assertFalse(self.panel.waist_participation.isEnabled())
        self.panel.process = None
        self.assertFalse(self.node.hand_commands)
        self.panel._execute_after_plan = True
        self.panel.stop()
        self.assertFalse(self.panel._execute_after_plan)

    def test_direct_execution_failed_precheck_never_starts_runner(self):
        self.panel.set_observation(self.obs)
        self.panel.scene_confirm.setChecked(True)
        with patch.object(self.panel, 'launch'):
            self.panel.execute_button.click()
        self.panel.job_directory = Path(self.tmp.name)
        (self.panel.job_directory/'result.json').write_text(json.dumps({'ok':False,'error':'IK rejected'}))
        self.panel.process = SimpleNamespace(deleteLater=lambda:None)
        with patch.object(self.panel, 'read_output'), patch.object(self.panel, '_execute_ready_plan') as execute:
            self.panel.finished(1,0)
            execute.assert_not_called()
        self.assertEqual(self.panel.sequence.state,'failed')
        self.assertIn('执行未启动：IK 预检失败', self.panel.message)
        self.assertIn('IK rejected', self.panel.execution_feedback.text())
        self.assertFalse(self.panel._execute_after_plan)
        self.assertIsNone(self.panel.runner)
        self.assertFalse(self.node.hand_commands)

    def test_execute_button_names_the_active_backend(self):
        self.panel.set_observation(self.obs)
        self.assertIn('真机', self.panel.execute_button.text())
        self.panel.execution_mode.setCurrentIndex(0)
        self.assertIn('仿真', self.panel.execute_button.text())

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
