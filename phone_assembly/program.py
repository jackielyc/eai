"""Backend-neutral assembly program. Completion requires object-state predicates."""
from __future__ import annotations
import copy
import numpy as np


STAGES = ('拿起手机','放入手机盒','盖上手机盒')
# key, label, high-level stage. Targets are resolved from observations at each action.
STEPS = (
    ('open_phone','张开手准备抓取',0),('above_phone','移动到手机上方',0),
    ('grasp_phone','下降到手机抓取位',0),('close_phone','夹持手机',0),
    ('check_grip_phone','检查手机双侧接触',0),('lift_phone','抬起手机',0),
    ('check_lift_phone','验证手机随手抬起',0),
    ('above_box_phone','携手机移动到盒上方',1),('place_phone','将手机放入盒内',1),
    ('release_phone','小开口释放手机',1),('retreat_phone','沿盒内间隙撤离',1),
    ('settle_phone','等待手机稳定',1),('check_phone','验证手机位于盒内',1),
    ('open_lid','张开手准备抓取盒盖',2),('above_lid','移动到盒盖上方',2),
    ('grasp_lid','下降到盒盖抓取位',2),('close_lid','夹持盒盖',2),
    ('check_grip_lid','检查盒盖双侧接触',2),('lift_lid','抬起盒盖',2),
    ('check_lift_lid','验证盒盖随手抬起',2),('above_box_lid','盒盖与盒体对齐',2),
    ('place_lid','下降盖合',2),('release_lid','释放盒盖',2),
    ('retreat_lid','撤离并观察',2),('settle_lid','等待盖合稳定',2),
    ('check_final','验收手机入盒与盒盖盖合',2),
)


def desired_object_pose(role,box,config):
    stored=config.get('placement_T_reference_object',{}).get(role)
    if stored is not None:return box@np.asarray(stored,float)
    if config.get('source')!='simulation_example':raise ValueError('真机缺少实测物体放置变换：'+role)
    desired=box.copy()
    desired[:3,3]+=box[:3,:3]@np.array([0,0,config['wall']+
        (config['phone_size'][2]/2 if role=='phone' else config['box_inner_size'][2]+config['lid_wall']/2)])
    return desired


def evaluate_goal(observation,config):
    objects=observation['objects'];box=np.asarray(objects['box'])
    cavity=np.asarray(config.get('T_box_frame_cavity') if config.get('T_box_frame_cavity') is not None else np.eye(4),float).copy()
    if config.get('T_box_frame_cavity') is None:cavity[2,3]=config['wall']
    T_cavity=box@cavity
    phone=np.linalg.inv(T_cavity)@np.asarray(objects['phone'])
    lid=np.linalg.inv(desired_object_pose('lid',box,config))@np.asarray(objects['lid'])
    phone_rotation=desired_object_pose('phone',box,config)[:3,:3].T@np.asarray(objects['phone'])[:3,:3]
    phone_angle=float(np.arccos(np.clip((np.trace(phone_rotation)-1)/2,-1,1)))
    oriented=phone_angle<.08
    half=np.abs(phone[:3,:3])@(np.asarray(config['phone_size'])/2)
    inner=np.asarray(config['box_inner_size']);wall=config['wall']
    contained=bool(np.all(abs(phone[:2,3])+half[:2] <= inner[:2]/2+.0005)
                   and phone[2,3]-half[2]>=-.002
                   and phone[2,3]+half[2]<=inner[2]+.001)
    target_lid_z=0.
    angle=float(np.arccos(np.clip((np.trace(lid[:3,:3])-1)/2,-1,1)))
    lid_error=float(np.linalg.norm(lid[:2,3]))
    covered=bool(lid_error<.002 and abs(lid[2,3]-target_lid_z)<.002 and angle<.05)
    velocities=observation.get('object_velocities',{})
    angular=observation.get('object_angular_velocities',{})
    stable=all(velocities.get(name) is not None and angular.get(name) is not None and
               np.isfinite([velocities[name],angular[name]]).all() and velocities[name]<.01 and angular[name]<.05
               for name in ('phone','lid'))
    released=not any(observation.get('contact_objects',{}).get(name,False) for name in ('phone','lid'))
    tool_clear=float(np.dot(np.asarray(observation['tool_pose'])[:3,3]-np.asarray(objects['lid'])[:3,3],T_cavity[:3,2]))>.05
    return {'phone_inside':contained,'phone_oriented':oriented,'phone_angle_rad':phone_angle,'lid_closed':covered,'stable':stable,
            'phone_relative_xyz':phone[:3,3].tolist(),'lid_relative_xyz':lid[:3,3].tolist(),
            'lid_xy_error_m':lid_error,'lid_height_error_m':float(lid[2,3]-target_lid_z),
            'lid_angle_rad':angle,'released':released,'tool_clear':tool_clear,
            'success':bool(contained and oriented and covered and stable and released and tool_clear)}


class AssemblyProgram:
    def __init__(self,backend,config):
        self.backend=backend;self.config=config;self.index=0
        self.state='idle';self.message='场景就绪，等待开始';self.events=[]
        self.offsets={};self.last_place={};self.pending=False
        self.preparing=False
        self.verify_started=None;self.stable_checks=0
        self.last_observation=None
    @property
    def step(self):return STEPS[self.index] if self.index<len(STEPS) else None
    def start(self):
        if self.state!='idle':raise ValueError('当前任务不是初始状态，请复位后重试')
        self.state='running';self.message='开始手机盒装配'
    def pause(self):
        if self.state=='running':
            if hasattr(self.backend,'pause'):self.backend.pause()
            self.state='paused';self.message='已请求暂停，等待执行器保持'
    def resume(self):
        if self.state=='paused':
            if hasattr(self.backend,'resume'):self.backend.resume()
            self.state='running'
    def cancel(self):
        self.backend.cancel();self.state='cancelled';self.message='已取消，保留当前场景供检查'
    def _target(self,key,obs):
        role='phone' if 'phone' in key else 'lid'
        obj=np.asarray(obs['objects'][role]);tool=np.asarray(obs['tool_pose'])
        if key.startswith(('above_box','place')):
            box=np.asarray(obs['objects']['box']);desired=desired_object_pose(role,box,self.config)
            desired[:3,3]+=box[:3,:3]@np.array([0,0,self.config.get('release_height_m',.003)])
            target=desired@np.linalg.inv(self.offsets[role])
            self.last_place[role]=target.copy()
            if key.startswith('above_box'):target[2,3]+=self.config['clearance_m']
        elif key.startswith('retreat'):
            target=self.last_place[role].copy();target[2,3]+=self.config['clearance_m']
        elif key.startswith('lift'):
            target=tool.copy();target[2,3]+=self.config['clearance_m']
        else:
            grasp=self.config.get('grasp_T_object_tool',{}).get(role)
            if grasp is not None:target=obj@np.asarray(grasp,float)
            else:
                if self.config.get('source')!='simulation_example':raise ValueError('真机缺少实测抓取变换：'+role)
                target=obj.copy()
                bottom=self.config['phone_size'][2]/2 if role=='phone' else self.config['lid_skirt']+self.config['lid_wall']/2
                offset=.02-bottom+.0005  # Keep proxy fingertips 0.5mm above the support plane.
                target[:3,3]+=obj[:3,:3]@np.array([0,0,offset])
            if key.startswith('above'):target[2,3]+=self.config['clearance_m']
        return target
    def _action(self,key,obs):
        role='phone' if 'phone' in key else 'lid'
        width=self.config['phone_size'][1] if role=='phone' else self.config['lid_inner_size'][1]+2*self.config['lid_wall']
        if key.startswith(('open','close','release')):
            action={'kind':'hand','open':not key.startswith('close'),'width':width,'object':role}
            if key.startswith('release'):action['gap_m']=width+.004
            return action
        if key.startswith('settle'):return {'kind':'wait','duration':self.config['settle_s']}
        return {'kind':'move','pose':self._target(key,obs).tolist(),'object':role,'task_step':key,
                'reference':'box' if key.startswith(('above_box','place','retreat')) else role,
                'speed':self.config['approach_speed_m_s'] if key.startswith(('grasp','place','retreat')) else self.config['tool_speed_m_s']}
    def _verify(self,key,obs):
        role='phone' if 'phone' in key else 'lid'
        if key.startswith('check_grip'):
            if obs['contact_objects'].get(role):
                self.offsets[role]=np.linalg.inv(np.asarray(obs['tool_pose']))@np.asarray(obs['objects'][role])
                return True
            return False
        if key.startswith('check_lift'):
            T=np.asarray(obs['objects'][role]);relative=np.linalg.inv(np.asarray(obs['tool_pose']))@T
            angle=float(np.arccos(np.clip((np.trace(relative[:3,:3].T@self.offsets[role][:3,:3])-1)/2,-1,1)))
            return bool(T[2,3]>self.config['table_z']+.05 and
                        np.linalg.norm(relative[:3,3]-self.offsets[role][:3,3])<.008 and angle<.12 and obs['contact_objects'].get(role))
        result=evaluate_goal(obs,self.config)
        if key=='check_phone':
            velocity=obs.get('object_velocities',{}).get('phone')
            angular=obs.get('object_angular_velocities',{}).get('phone')
            return result['phone_inside'] and result['phone_oriented'] and velocity is not None and angular is not None and velocity<.01 and angular<.05
        return result['success']
    def tick(self,dt=.02):
        if self.state!='running':return
        try:
            self.backend.tick(dt)
            if getattr(self.backend,'pause_required',False):
                self.pause();self.message=self.backend.pause_reason or '执行器要求暂停';return
            if self.step is None:
                self.state='complete';self.message='验收通过：手机位于盒内，盒盖已盖合且物体稳定';return
            key,label,group=self.step;obs=self.backend.observation();self.last_observation=copy.deepcopy(obs);self.message=label
            if key.startswith('check'):
                if self.verify_started is None:
                    self.verify_started=obs['sim_time']
                    if hasattr(self.backend,'begin_verification'):self.backend.begin_verification(key)
                ready=obs.get('verification_ready',True)
                self.stable_checks=self.stable_checks+1 if ready and self._verify(key,obs) else 0
                if self.stable_checks>=5:self._advance(obs)
                elif obs['sim_time']-self.verify_started>self.config.get('verify_timeout_s',3):raise ValueError(label+'未通过：'+str(evaluate_goal(obs,self.config)))
                return
            if not self.pending:
                if hasattr(self.backend,'prepare_action'):
                    if not self.preparing:self.backend.prepare_action(key);self.preparing=True
                    if not self.backend.preparation_ready:return
                    obs=self.backend.observation()
                action=self._action(key,obs);self.backend.start(action);self.pending=True
                self.events.append({'event':'action_started','step':key,'time':obs['sim_time'],'action':action})
            elif self.backend.action_state=='complete':self._advance(obs)
            elif self.backend.action_state in ('failed','cancelled'):raise ValueError(self.backend.action_error or label+'失败')
        except Exception as exc:
            self.backend.cancel();self.state='failed';self.message=str(exc)
            self.events.append({'event':'failed','step':self.step[0] if self.step else '', 'message':self.message})
    def _advance(self,obs):
        if self.step[0].startswith('check_lift'):
            role='phone' if 'phone' in self.step[0] else 'lid'
            self.offsets[role]=np.linalg.inv(np.asarray(obs['tool_pose']))@np.asarray(obs['objects'][role])
        if hasattr(self.backend,'on_step_completed'):self.backend.on_step_completed(self.step[0],obs)
        self.events.append({'event':'step_completed','step':self.step[0],'time':obs['sim_time']})
        self.index+=1;self.pending=False;self.preparing=False;self.verify_started=None;self.stable_checks=0
        if self.index>=len(STEPS):
            self.state='complete';self.message='验收通过：手机位于盒内，盒盖已盖合且物体稳定'
    def status(self):
        try:obs=self.backend.observation();self.last_observation=copy.deepcopy(obs)
        except Exception:
            obs=copy.deepcopy(self.last_observation)
            if obs is None:
                return {'state':self.state,'message':self.message,'index':self.index,'total':len(STEPS),
                        'group':self.step[2] if self.step else len(STAGES),'observation':{},'goal':{'success':False},'events':self.events[-5:]}
            obs['stale']=True
        goal=evaluate_goal(obs,self.config)
        if obs.get('stale'):goal['success']=False
        return {'state':self.state,'message':self.message,'index':self.index,'total':len(STEPS),
                'group':self.step[2] if self.step else len(STAGES),'step':self.step[0] if self.step else None,
                'observation':obs,'goal':goal,'events':self.events[-5:]}
