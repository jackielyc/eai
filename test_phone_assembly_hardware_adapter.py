"""Synthetic calibration + fake transport only; never instantiate a ROS adapter."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
import numpy as np
from phone_assembly.scene import scenario
from phone_assembly.hardware_backend import HardwareAssemblyBackend
from grasp_world import ManipulationWorld


class Value:
    def __init__(self,value):self.item=value
    def currentData(self):return self.item
    def isChecked(self):return bool(self.item)
    def value(self):return self.item


class HardwareAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name)
        cfg=scenario();cfg.update(source='hardware_calibration',hardware_calibrated=True,tool_model='ruiyan_6dof_v1',
            arm_side='right',pose_method='pca',T_link_tool=np.eye(4).tolist(),T_box_frame_cavity=np.eye(4).tolist(),
            placement_T_reference_object={k:np.eye(4).tolist() for k in ('phone','lid')},
            grasp_T_object_tool={k:np.eye(4).tolist() for k in ('phone','lid')},
            hand_presets={k:{'open':[0.]*6,'close':[.7]*6,'release':[.2]*6,'velocity':[50.]*6,'effort':[100.]*6} for k in ('phone','lid')})
        cfg['T_link_tool'][2][3]=.1
        self.config=root/'hardware.json';self.config.write_text(json.dumps(cfg))
        assembly=root/'assembly.json';assembly.write_text('{"relations":{}}')
        self.calls=[];self.vision=[]
        profile={'hand_model':'ruiyan_6dof_v1','tcp_link':'Link7_r','arm_side':'right',
                 'force_policy':{'calibrated':True,'T_mount_sensor':np.eye(4).tolist(),'bias':[0]*6,'component_limits':[15]*3+[1]*3}}
        targets={}
        for role in ('phone','box','lid'):
            T=np.eye(4);T[:3,3]=[.5,-.2,.73]
            obs={'observation_id':role+'-1','captured_at':1.,'base_frame':'base_link','T_base_camera':np.eye(4).tolist(),
                 'T_base_object':T.tolist(),'object_pose':{'T_camera_object':T.tolist(),'extents_m':[.1,.07,.01]},
                 'provenance':{'version':1,'issues':[]}}
            targets[role]={'methods':{'pca':obs}}
        self.panel=NS(is_simulation=False,assembly_path=assembly,runtime_scene=lambda:{'world':[]},
            feedback=lambda:{'q16':[0.]*16,'right':{'xyz':[.5,-.2,.8],'xyzw':[0,0,0,1]},'urdf_sha256':'synthetic'},
            runtime_hands=lambda:{'left':[0]*6,'right':[0]*6},scene_confirm=Value(True),wrench_guard_check=Value(True),
            selected_profile=lambda:copy.deepcopy(profile),world=ManipulationWorld(),_unverified_grips=set(),
            grasp_arm=Value('right'),source_combo=Value('pca'),task_role_combos={k:Value(k) for k in targets},
            target_observations=targets,reestimate_callback=lambda role,cb:self.vision.append((role,cb)),
            _program_backend_authorized=False,_program_active=True,runner=None,process=None,hand_runner=None,
            force_limit=Value(15),torque_limit=Value(1),waist_participation=Value(True),runtime_plan_inputs=lambda:{},
            wrench_feedback=lambda side:{'stamp':time.time(),'force':[0]*3,'torque':[0]*3},
            record_workflow_event=lambda *a,**kw:None,message='')
        self.panel.reserve=lambda **kw:self.calls.append(('reserve',self.panel._program_backend_authorized))
        self.panel.launch=lambda mode,**kw:self.calls.append(('plan',kw))
        self.panel.new_process_runner=lambda kind,**kw:self.calls.append(('hand',kw)) or NS(state='running')
        self.panel.stop=lambda:self.calls.append(('stop',self.panel._program_backend_authorized))
        self.backend=HardwareAssemblyBackend(self.panel,self.config)
    def tearDown(self):self.tmp.cleanup()
    def test_tool_goal_is_converted_to_link7_and_uses_existing_preflight(self):
        T=np.eye(4);T[:3,3]=[.6,-.1,.85]
        self.backend.start({'kind':'move','pose':T.tolist(),'object':'phone','reference':'box','task_step':'place_phone','speed':.02})
        goal=np.asarray(self.panel.sequence.preview['goals']['approach'])
        np.testing.assert_allclose(goal[:3,3],[.6,-.1,.75])
        self.assertTrue(self.panel.sequence.preview['program_goal'])
        self.assertEqual(self.calls[0],('reserve',True))
        self.assertEqual(self.calls[-1][1]['stage'],'approach')
        self.assertEqual(self.calls[-1][1]['cartesian_speed_m_s'],.02)
        self.assertFalse(self.panel._program_backend_authorized)
    def test_release_uses_measured_partial_open_not_full_open(self):
        self.backend.start({'kind':'hand','open':True,'gap_m':.074,'width':.07,'object':'phone'})
        self.assertEqual(self.calls[-1][1]['target'],[.2]*6)
        self.assertEqual(self.calls[-1][1]['hand_operation'],'hand_open')
        self.assertEqual(self.calls[-1][1]['hand_velocity'],[50.]*6)
    def test_late_visual_result_after_cancel_is_ignored(self):
        self.backend.prepare_action('grasp_phone')
        before=self.backend.observations['phone']['observation_id'];callback=self.vision[0][1]
        self.backend.cancel()
        obs=copy.deepcopy(self.backend.observations['phone']);obs.update(observation_id='late',captured_at=2.)
        callback({'pca':obs})
        self.assertEqual(self.backend.observations['phone']['observation_id'],before)
        self.assertFalse(any(kind=='plan' for kind,_ in self.calls))


if __name__=='__main__':unittest.main()
