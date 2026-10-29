import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from click_pose import FrameBuffer
from grasp_trajectory import limits_from_profile,parameterize,extrema,sample,exceeds
from grasp_perception import capture_provenance,verify_live,require_verified,mark_cached
from grasp_world import ManipulationWorld,expand_skill
from grasp_candidates import orientation_proposals


class TrajectoryTests(unittest.TestCase):
    def test_mount_lever_arm_is_included_without_disabling_raw_force_limit(self):
        from grasp_execution import evaluate_wrench
        T=np.eye(4);T[2,3]=.1
        policy={'calibrated':True,'T_mount_sensor':T.tolist(),'bias':[0]*6,
                'component_limits':[100,100,100,100,.5,100]}
        sample={'stamp':100.,'force':[10.,0,0],'torque':[0.,0,0]}
        error,*_=evaluate_wrench(sample,100,100,now=100.,policy=policy)
        self.assertIn('方向分量',error)
        policy['bias']=[10.,0,0,0,0,0]
        error,*_=evaluate_wrench(sample,5,100,now=100.,policy=policy)
        self.assertIn('合力',error)
    def test_continuous_velocity_and_per_joint_jerk_limits(self):
        rows=np.zeros((4,16));rows[:,0]=[0,.01,.02,.03];rows[:,4]=[0,.1,.2,.3]
        limits=limits_from_profile({'joint_velocity_rad_s':.1,'joint_acceleration_rad_s2':.2})
        plan=dict(parameterize(rows,limits),joint_waypoints=rows.tolist())
        self.assertTrue(plan['trajectory_blended'])
        prior_velocity=None;prior_acceleration=None
        for i,c in enumerate(np.asarray(plan['trajectory_coefficients'])):
            dt=plan['times_s'][i+1]-plan['times_s'][i]
            for derivative,name in enumerate(('velocity','acceleration','jerk'),1):
                self.assertTrue(np.all(np.max(abs(extrema(c,derivative)),axis=1)/dt**derivative<=limits[name]+1e-9))
            if prior_velocity is not None:
                np.testing.assert_allclose(c[1]/dt,prior_velocity,atol=1e-10)
                np.testing.assert_allclose(2*c[2]/dt**2,prior_acceleration,atol=1e-10)
            prior_velocity=np.sum(np.arange(1,6)[:,None]*c[1:],axis=0)/dt
            prior_acceleration=np.sum((np.arange(2,6)*np.arange(1,5))[:,None]*c[2:],axis=0)/dt**2
        self.assertGreater(abs(np.asarray(plan['trajectory_coefficients'])[1,1,4]),0)
        np.testing.assert_allclose(sample(plan,plan['times_s'][-1]),rows[-1])

    def test_lift_and_revolute_tracking_use_different_units(self):
        q=np.zeros(16);q[0]=.03
        self.assertTrue(exceeds(q,np.zeros(16),'tracking'))
        q[0]=0;q[2]=.03
        self.assertFalse(exceeds(q,np.zeros(16),'tracking'))


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'calibration.json'
        self.path.write_text('{"cameras":{"color":{"depth_registered_to_color":true}}}')
        self.buffer=FrameBuffer();self.image=np.zeros((64,64,3),np.uint8)
        self.buffer.add('color',self.image,10.,'camera','bgr8')
        self.buffer.add('depth',np.ones((64,64)),10.02,'camera','32FC1')
        self.provenance=capture_provenance(self.buffer,{'head':('color','depth',self.image)},self.path,'sensor')
        self.observation={'observation_id':'a','provenance':self.provenance}
    def tearDown(self):self.tmp.cleanup()
    def test_cache_remains_previewable_but_requires_fresh_evidence(self):
        mark_cached({'pca':self.observation})
        with self.assertRaises(ValueError):require_verified(self.observation,None)
        self.buffer.add('color',self.image.copy(),11.,'camera','bgr8')
        result=verify_live(self.observation,self.buffer,self.path)
        require_verified(self.observation,result)
        self.assertFalse(result['pose_reestimated'])
        with self.assertRaises(ValueError):require_verified({'observation_id':'other'},result)
    def test_changed_scene_or_calibration_is_rejected(self):
        self.buffer.add('color',self.image+100,11.,'camera','bgr8')
        with self.assertRaisesRegex(ValueError,'图像变化'):verify_live(self.observation,self.buffer,self.path)
        self.path.write_text('{"revision":2}')
        with self.assertRaisesRegex(ValueError,'标定已变化'):verify_live(self.observation,self.buffer,self.path)
    def test_lingbot_uses_color_provenance_without_inventing_sensor_depth(self):
        result=capture_provenance(self.buffer,{'head':('color','missing',self.image)},self.path,'lingbot')
        self.assertEqual(result['issues'],[])
        self.assertNotIn('depth_stamp',result['frames']['head'])
    def test_unsynchronized_depth_blocks_execution_evidence(self):
        buffer=FrameBuffer();buffer.add('color',self.image,10,'camera','bgr8')
        buffer.add('depth',np.ones((64,64)),11,'camera','32FC1')
        result=capture_provenance(buffer,{'head':('color','depth',self.image)},self.path,'sensor')
        self.assertTrue(result['issues'])


class WorldTests(unittest.TestCase):
    def test_joint_transit_routes_around_obstacle(self):
        from grasp_rrt import plan_transit
        start=np.zeros(16);goal=start.copy();goal[2]=1
        def check(q):
            if .35<q[2]<.65 and abs(q[3])<.2:raise ValueError('obstacle')
        rows=plan_transit(start,goal,np.full(16,-1.),np.full(16,1.),[2,3],check,seconds=2)
        np.testing.assert_allclose(rows[0],start);np.testing.assert_allclose(rows[-1],goal)
        self.assertGreater(max(abs(q[3]) for q in rows),.2)
        for a,b in zip(rows,rows[1:]):
            for t in np.linspace(0,1,10):check(np.array(a)*(1-t)+np.array(b)*t)
    def test_place_uses_object_to_tcp_transform_and_needs_calibration(self):
        world=ManipulationWorld();T=np.eye(4);T[0,3]=1.
        obs={'T_base_object':T.tolist(),'T_base_camera':np.eye(4).tolist(),
             'object_pose':{'T_camera_object':T.tolist(),'extents_m':[.1,.05,.02]}}
        feedback={'right':{'xyz':[.9,0,0],'xyzw':[0,0,0,1]}}
        world.attach_confirmed('right','phone','phone-id',obs,feedback)
        ref=np.eye(4);ref[1,3]=2.;reference=dict(obs,T_base_object=ref.tolist())
        preview={'id':'p','goals':{}}
        with self.assertRaisesRegex(ValueError,'待标定'):
            world.placement_preview(preview,'task','right','box',reference,{'relations':{}})
        relation=np.eye(4);relation[0,3]=.2
        result=world.placement_preview(preview,'task','right','box',reference,
            {'relations':{'task:phone:box':{'calibrated':True,'T_reference_object':relation.tolist()}}})
        np.testing.assert_allclose(np.array(result['goals']['approach'])[:3,3],[.1,2.,0],atol=1e-10)
        self.assertEqual(len(world.attached),1)
        world.detach('right',feedback)
        self.assertFalse(world.attached);self.assertIn('phone-id',world.placed)
    def test_skill_expansion_keeps_explicit_result_checks(self):
        for kind in ('pick','place','align','insert'):
            result=expand_skill({'label':'test','kind':kind,'role':'object'})
            self.assertEqual(result['kind'],'sequence')
            self.assertIn('confirm',[a['kind'] for a in result['actions']])
    def test_proposals_preserve_contact_position(self):
        T=np.eye(4);T[:3,3]=[.5,.1,.8]
        result=orientation_proposals({'T_camera_grasp':T.tolist()})
        self.assertEqual(len(result),3)
        for item in result:
            np.testing.assert_allclose(np.asarray(item['T_camera_grasp'])[:3,3],T[:3,3])


if __name__=='__main__':unittest.main()
