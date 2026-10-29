"""Dynamic object contact backend; object qpos is only initialized by reset."""
from __future__ import annotations
import numpy as np
from .scene import xml_scene


class MujocoAssemblyBackend:
    def __init__(self, config):
        if config.get('tool_model')!='parallel_jaw_contact_proxy' or config.get('source')!='simulation_example':
            raise ValueError('此后端只接受明确标记的接触代理仿真配置，不能冒充真实睿研手模型')
        import mujoco
        self.mj=mujoco;self.config=config
        self.model=mujoco.MjModel.from_xml_string(xml_scene(config));self.data=mujoco.MjData(self.model)
        self.target_id=self.model.body('tool_target').mocapid[0]
        for name in ('jaw_left_slide','jaw_right_slide'):
            self.data.qpos[self.model.joint(name).qposadr[0]]=.06
        self.data.ctrl[:]=.06
        self.mj.mj_forward(self.model,self.data)
        self.lid_geom_ids=[i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]==self.model.body('lid').id]
        self.lid_alpha={i:float(self.model.geom_rgba[i,3]) for i in self.lid_geom_ids}
        self.action=None;self.action_state='idle';self.action_error=''
        self.held_relative=None;self.object_write_count=0
        self.maximum_tool_force=0.;self.maximum_tool_torque=0.
        self.simulation_valid=True
        for _ in range(250):self._physics_step()
        self._last_valid_observation=self.observation()

    def _physics_step(self):
        before=float(self.data.time);self.mj.mj_step(self.model,self.data)
        bad=any(self.data.warning[int(getattr(self.mj.mjtWarning,name))].number for name in
                ('mjWARN_BADQPOS','mjWARN_BADQVEL','mjWARN_BADQACC','mjWARN_BADCTRL'))
        if bad or self.data.time<before or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.simulation_valid=False;self.action_error='MuJoCo数值异常或自动重置，本次物理结果无效'
            raise ValueError(self.action_error)

    @property
    def time(self):return float(self.data.time)

    def pose(self,name):
        body=self.data.body(name);T=np.eye(4)
        T[:3,3]=body.xpos;T[:3,:3]=body.xmat.reshape(3,3)
        return T

    def observation(self):
        if not self.simulation_valid and hasattr(self,'_last_valid_observation'):
            return dict(self._last_valid_observation,simulation_valid=False,stale=True)
        result={'objects':{name:self.pose(name).tolist() for name in ('phone','box','lid')},
                'tool_pose':self.pose('tool').tolist(),'sim_time':self.time,
                'object_velocities':{name:float(np.linalg.norm(self.data.body(name).cvel[3:])) for name in ('phone','lid')},
                'object_angular_velocities':{name:float(np.linalg.norm(self.data.body(name).cvel[:3])) for name in ('phone','lid')},
                'contact_objects':{name:self.grasp_contact(name) for name in ('phone','lid')},
                'maximum_tool_force_n':self.maximum_tool_force,'maximum_tool_torque_nm':self.maximum_tool_torque,
                'source':'simulation_ground_truth'}
        self._last_valid_observation=result
        return result

    def tool_wrench(self):
        ids={self.model.body(name).id for name in ('tool','jaw_left','jaw_right')}
        force=np.zeros(3);torque=np.zeros(3);origin=self.pose('tool')[:3,3]
        for i in range(self.data.ncon):
            c=self.data.contact[i];a=self.model.geom_bodyid[c.geom1] in ids;b=self.model.geom_bodyid[c.geom2] in ids
            if a==b:continue
            local=np.zeros(6);self.mj.mj_contactForce(self.model,self.data,i,local)
            sign=1 if b else -1;R=c.frame.reshape(3,3).T
            f=sign*(R@local[:3]);force+=f;torque+=sign*(R@local[3:])+np.cross(c.pos-origin,f)
        return float(np.linalg.norm(force)),float(np.linalg.norm(torque))

    def grasp_contact(self,name):
        body_id=self.model.body(name).id;contacts=set()
        for i in range(self.data.ncon):
            c=self.data.contact[i]
            ids=(c.geom1,c.geom2)
            for a,b in (ids,ids[::-1]):
                if self.model.geom_bodyid[a]==body_id:
                    other=self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_GEOM,b)
                    if other in ('jaw_left_pad','jaw_right_pad'):contacts.add(other)
        return len(contacts)==2

    def start(self,action):
        if self.action_state=='running':raise ValueError('已有动作在运行')
        self.action=dict(action);self.action_state='running';self.action_error='';self.started=self.time
        self.stable=0
        if action['kind']=='move':
            self.start_pose=self.data.mocap_pos[self.target_id].copy()
            T=np.asarray(action['pose'],float)
            self.goal=T[:3,3]
            self.start_quat=self.data.mocap_quat[self.target_id].copy()
            self.goal_quat=np.empty(4);self.mj.mju_mat2Quat(self.goal_quat,T[:3,:3].reshape(-1))
            self.duration=max(.25,float(np.linalg.norm(self.goal-self.start_pose))/action.get('speed',self.config['tool_speed_m_s']))
        elif action['kind']=='hand':
            self.hand_start=self.data.ctrl.copy();self.hand_duration=.7
            if action['open']:self.hand_goal=np.full(2,min(.06,action.get('gap_m',.117)/2+.0015))
            else:
                width=action['width'];self.hand_goal=np.full(2,width/2+.0015-.005)
        elif action['kind']!='wait':raise ValueError('未知仿真动作')

    def tick(self,dt=.02):
        for _ in range(max(1,int(round(dt/self.model.opt.timestep)))):
            if self.action_state=='running' and self.action['kind']=='move':
                u=min(1.,(self.time-self.started)/self.duration);s=10*u**3-15*u**4+6*u**5
                self.data.mocap_pos[self.target_id]=self.start_pose+s*(self.goal-self.start_pose)
                target=self.goal_quat if np.dot(self.start_quat,self.goal_quat)>=0 else -self.goal_quat
                quat=(1-s)*self.start_quat+s*target
                self.data.mocap_quat[self.target_id]=quat/np.linalg.norm(quat)
            if self.action_state=='running' and self.action['kind']=='hand':
                u=min(1.,(self.time-self.started)/self.hand_duration);s=10*u**3-15*u**4+6*u**5
                self.data.ctrl[:]=self.hand_start+s*(self.hand_goal-self.hand_start)
            self._physics_step()
            force,torque=self.tool_wrench()
            self.maximum_tool_force=max(self.maximum_tool_force,force);self.maximum_tool_torque=max(self.maximum_tool_torque,torque)
            if self.action_state=='running' and (force>self.config.get('force_limit_n',15.) or torque>self.config.get('torque_limit_nm',1.)):
                self.data.mocap_pos[self.target_id]=self.pose('tool')[:3,3]
                self.action_state='failed';self.action_error=f'仿真接触载荷超限：{force:.2f}N / {torque:.3f}Nm'
        if self.action_state!='running':return
        elapsed=self.time-self.started;action=self.action
        if action['kind']=='move':
            actual=self.pose('tool');target=np.asarray(action['pose'])
            angle=float(np.arccos(np.clip((np.trace(target[:3,:3].T@actual[:3,:3])-1)/2,-1,1)))
            done=elapsed>=self.duration and np.linalg.norm(actual[:3,3]-self.goal)<.003 and angle<.02
        elif action['kind']=='hand':
            if action['open']:
                done=all(abs(self.data.qpos[self.model.joint(n).qposadr[0]]-self.hand_goal[i])<.001
                         for i,n in enumerate(('jaw_left_slide','jaw_right_slide')))
            else:done=self.grasp_contact(action['object'])
            done=done and elapsed>=self.hand_duration
        else:done=elapsed>=action.get('duration',.5)
        self.stable=self.stable+1 if done else 0
        if self.stable>=5:self.action_state='complete'
        if elapsed>max(8.,getattr(self,'duration',0)+3) and self.action_state=='running':
            self.action_state='failed';self.action_error='动作未到位或未形成双侧夹持接触'

    def cancel(self):
        if self.simulation_valid:
            self.data.mocap_pos[self.target_id]=self.pose('tool')[:3,3]
            self.data.mocap_quat[self.target_id]=self.data.body('tool').xquat
        self.action_state='cancelled'

    def update_robot_visual(self,mapping):
        pin=mapping.pin
        pin.forwardKinematics(mapping.ik.model,mapping.ik.data,mapping.ik.q_from_list(mapping.q))
        pin.updateFramePlacements(mapping.ik.model,mapping.ik.data)
        for index,frame in enumerate(mapping.ik.model.frames):
            name='robot_'+frame.name
            body_id=self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_BODY,name)
            if body_id<0:continue
            mocap=self.model.body_mocapid[body_id]
            if mocap<0:continue
            T=mapping.ik.data.oMf[index]
            self.data.mocap_pos[mocap]=T.translation
            self.mj.mju_mat2Quat(self.data.mocap_quat[mocap],T.rotation.reshape(-1))
        self.mj.mj_forward(self.model,self.data)

    def close(self):
        if hasattr(self,'renderer'):self.renderer.close();del self.renderer

    def render(self,width=800,height=600,view='close'):
        for i in self.lid_geom_ids:self.model.geom_rgba[i,3]=.25 if view=='inspect' else self.lid_alpha[i]
        if not hasattr(self,'renderer'):
            self.renderer=self.mj.Renderer(self.model,height=height,width=width)
            self.camera=self.mj.MjvCamera();self.camera.lookat[:]=[.55,-.09,.75]
            self.camera.distance=.9;self.camera.azimuth=90;self.camera.elevation=-45
        if view=='inspect':
            self.camera.lookat[:]=[.55,-.10,.75];self.camera.distance=.6;self.camera.azimuth=0;self.camera.elevation=-55
        elif view=='robot':
            self.camera.lookat[:]=[.3,-.05,.85];self.camera.distance=1.8;self.camera.azimuth=140;self.camera.elevation=-25
        else:
            self.camera.lookat[:]=[.55,-.09,.75];self.camera.distance=.85;self.camera.azimuth=90;self.camera.elevation=-45
        self.renderer.update_scene(self.data,camera=self.camera)
        return self.renderer.render().copy()
