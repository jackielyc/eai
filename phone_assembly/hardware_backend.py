"""Adapter from the shared program to existing calibrated A2D planning/execution.

No simulated poses or motor mappings are accepted here. Grasp checkpoints require
operator evidence; placement checkpoints consume new visual estimates.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time
import numpy as np
from grasp_execution import Sequence,fingerprint
from grasp_world import object_matrix,tcp_matrix
from .config import load_hardware


class HardwareAssemblyBackend:
    def __init__(self,panel,config_path):
        if panel.is_simulation:raise ValueError('真机适配器不能在仿真模式下启动')
        self.panel=panel;self.config_path=Path(config_path)
        self.config=load_hardware(self.config_path,panel.assembly_path)
        self.config_hash=fingerprint(json.loads(self.config_path.read_text()))
        if not callable(panel.reestimate_callback):raise ValueError('未连接真机视觉重新估计接口')
        panel.runtime_scene();panel.feedback();panel.runtime_hands()
        if not panel.wrench_guard_check.isChecked() or not panel.selected_profile().get('force_policy',{}).get('calibrated'):
            raise ValueError('装配任务需要启用已标定的连接件六维力保护')
        if not panel.scene_confirm.isChecked():raise ValueError('请先核对现场与标定并勾选执行确认')
        if panel.world.attached or panel._unverified_grips:raise ValueError('开始新任务前需安全放下已有物体，并确认手为空')
        self.side=str(panel.grasp_arm.currentData());self.method=str(panel.source_combo.currentData())
        if self.config.get('arm_side')!=self.side:raise ValueError('当前手臂与任务实测TCP标定侧别不一致')
        if self.config.get('pose_method')!=self.method:raise ValueError('当前视觉方法与任务实测标定不一致')
        self.targets={};self.observations={};self.objects={};self.velocities={'phone':None,'lid':None}
        self.angular_velocities={'phone':None,'lid':None}
        self.contacts={'phone':False,'lid':False};self.role_for_hand=None
        for role in ('phone','box','lid'):
            target=str(panel.task_role_combos[role].currentData() or '')
            if not target:raise ValueError('请绑定真机目标：'+role)
            self.targets[role]=target
            obs=panel.target_observations[target]['methods'].get(self.method)
            if not obs:raise ValueError('目标缺少当前位姿方法：'+role)
            self._adopt(role,obs,require_live=False)
        self.action_state='idle';self.action_error='';self.action=None
        self.preparation_ready=True;self.verification_ready=True;self.needs_operator=False
        self.operator_confirmed=False;self.vision_pending=False;self.vision_queue=[];self.vision_generation=0
        self.cancelled=False;self.verify_key=None;self.vision_error=None
        self.pause_required=False;self.pause_reason='';self.box_verified_pose=None
        self.panel._program_validator=self.validate

    @contextmanager
    def authorized(self):
        old=getattr(self.panel,'_program_backend_authorized',False)
        self.panel._program_backend_authorized=True
        try:yield
        finally:self.panel._program_backend_authorized=old

    def validate(self):
        if fingerprint(json.loads(self.config_path.read_text()))!=self.config_hash:
            raise ValueError('装配真机标定文件已变化，请停止后重新启动任务')
        if self.panel.is_simulation:raise ValueError('执行模式变化，拒绝继续真机任务')
        if not self.panel.wrench_guard_check.isChecked():raise ValueError('装配任务不能关闭六维力保护')
        if str(self.panel.grasp_arm.currentData())!=self.side or str(self.panel.source_combo.currentData())!=self.method:
            raise ValueError('执行手臂或位姿方法变化，拒绝继续任务')

    def _adopt(self,role,observation,require_live=True):
        if observation.get('execution_blockers'):raise ValueError('；'.join(observation['execution_blockers']))
        if observation.get('base_frame')!='base_link':raise ValueError('目标尚未转换到机器人base_link')
        provenance=observation.get('provenance') or {}
        if require_live and provenance.get('version')!=1:raise ValueError('新估计缺少采集/标定证据')
        if provenance.get('issues'):raise ValueError('视觉证据不足：'+'；'.join(provenance['issues']))
        T=object_matrix(observation)
        if role=='box' and getattr(self,'verify_key',None)=='check_final' and self.box_verified_pose is not None:
            if np.linalg.norm(T[:3,3]-self.box_verified_pose[:3,3])>.002:
                raise ValueError('盖合期间盒体移动；手机被遮挡，必须重新检查，不能沿用入盒验收')
        old=self.observations.get(role)
        if old:
            dt=float(observation.get('captured_at',0))-float(old.get('captured_at',0))
            if dt<=0:raise ValueError('视觉结果没有更新采集时间')
            self.velocities[role]=float(np.linalg.norm(T[:3,3]-self.objects[role][:3,3])/dt)
            self.angular_velocities[role]=float(np.arccos(np.clip((np.trace(self.objects[role][:3,:3].T@T[:3,:3])-1)/2,-1,1))/dt)
        self.observations[role]=copy.deepcopy(observation);self.objects[role]=T
        target=self.targets.get(role)
        if target in self.panel.world.placed:self.panel.world.placed[target]['T_base_object']=T.tolist()

    def _request_views(self,roles):
        self.vision_queue=list(roles);self.vision_pending=True;self.vision_error=None
        self.vision_deadline=time.monotonic()+120;self.vision_generation+=1
        self._next_view(self.vision_generation)

    def _next_view(self,generation):
        if generation!=self.vision_generation or self.cancelled:return
        if not self.vision_queue:
            self.vision_pending=False;self.preparation_ready=True;return
        role=self.vision_queue.pop(0)
        def finished(observations,error=None):
            if generation!=self.vision_generation or self.cancelled:return
            try:
                if error or self.method not in (observations or {}):raise ValueError(error or '所选方法无新估计')
                self._adopt(role,observations[self.method])
                self.panel.target_observations[self.targets[role]]['methods']=copy.deepcopy(observations)
                self._next_view(generation)
            except Exception as exc:self.vision_error=str(exc);self.vision_pending=False
        self.panel.reestimate_callback(self.targets[role],finished)

    def prepare_action(self,key):
        self.validate();self.preparation_ready=True
        if key=='open_phone':
            self.preparation_ready=False;self._request_views(['phone','box','lid']);return
        if key.startswith('retreat'):
            role='phone' if 'phone' in key else 'lid'
            self.preparation_ready=False;self._request_views([role,'box']);return
        if key.startswith(('above_phone','grasp_phone','above_lid','grasp_lid','above_box','place')):
            role='box' if key.startswith(('above_box','place')) else ('phone' if 'phone' in key else 'lid')
            self.preparation_ready=False;self._request_views([role])

    def begin_verification(self,key):
        self.verify_key=key;self.verification_ready=False;self.operator_confirmed=False
        self.needs_operator=key.startswith(('check_grip','check_lift'))
        if key.startswith(('check_grip','check_lift')):
            role='phone' if 'phone' in key else 'lid';self.contacts[role]=False
            self._request_views([role])
        elif key=='check_phone':self._request_views(['phone','box','phone','box'])
        else:self._request_views(['lid','box','lid','box'])

    def confirm_verification(self):
        if not self.needs_operator or self.vision_pending or self.vision_error:return
        self.operator_confirmed=True;role='phone' if 'phone' in self.verify_key else 'lid'
        self.contacts[role]=True
        self.panel.record_workflow_event('assembly_operator_verification',checkpoint=self.verify_key)

    def on_step_completed(self,key,observation):
        if key=='check_phone':self.box_verified_pose=np.asarray(observation['objects']['box']).copy()
        if key.startswith(('check_grip','check_lift')):
            role='phone' if 'phone' in key else 'lid'
            self.panel.world.attach_confirmed(self.side,role,self.targets[role],self.observations[role],
                self.panel.feedback(),replace=key.startswith('check_lift'))
            self.panel._unverified_grips.discard(self.side)

    def observation(self):
        T_tool=tcp_matrix(self.panel.feedback(),self.side)@np.asarray(self.config['T_link_tool'])
        self.verification_ready=not self.vision_pending and not self.vision_error and (not self.needs_operator or self.operator_confirmed)
        return {'objects':{k:v.tolist() for k,v in self.objects.items()},'tool_pose':T_tool.tolist(),
                'object_velocities':dict(self.velocities),'object_angular_velocities':dict(self.angular_velocities),
                'contact_objects':dict(self.contacts),
                'sim_time':time.monotonic(),'source':'hardware_vision_and_operator',
                'verification_ready':self.verification_ready}

    def start(self,action):
        self.validate();self.action=dict(action);self.action_state='running';self.action_error='';self.started=time.monotonic()
        p=self.panel;p._program_action_result=None
        if action['kind']=='wait':return
        role=action.get('reference') or action.get('object')
        p._program_observation_role=role;p.observation=copy.deepcopy(self.observations[role])
        with self.authorized():
            p.reserve(require_preview=False)
            if action['kind']=='hand':
                role=action['object'];key='close' if not action['open'] else ('release' if 'gap_m' in action else 'open')
                p._hand_purpose='program';self.role_for_hand=role
                p.hand_runner=p.new_process_runner('hand',side=self.side,target=self.config['hand_presets'][role][key],
                    hand_operation='hand_open' if action['open'] else 'hand_close',
                    hand_velocity=self.config['hand_presets'][role]['velocity'],hand_effort=self.config['hand_presets'][role]['effort'])
                return
            profile=p.selected_profile()
            T=np.asarray(action['pose'])@np.linalg.inv(np.asarray(self.config['T_link_tool']))
            stage='approach' if action['task_step'].startswith(('grasp','place','retreat')) else 'pregrasp'
            preview={'id':fingerprint({'step':action['task_step'],'goal':T.tolist(),'observation':p.observation['observation_id']}),
                'program_goal':True,
                'arm_side':self.side,'hand_model':profile['hand_model'],'base_frame':'base_link','tcp_link':profile['tcp_link'],
                'observation_id':p.observation['observation_id'],'goals':{k:T.tolist() for k in ('pregrasp','approach','lift')}}
            p.sequence=Sequence();p.sequence.reset(preview,profile);p.sequence.begin_plan(stage)
            p._execute_after_plan=True
            p.launch('plan',stage=stage,start_joints=p.feedback()['q16'],allow_waist=p.waist_participation.isChecked(),
                cartesian_speed_m_s=action['speed'],tool_offset_link_m=np.asarray(self.config['T_link_tool'])[:3,3].tolist(),
                **p.runtime_plan_inputs())

    def tick(self,dt=.02):
        if self.cancelled:return
        from grasp_execution import evaluate_wrench
        sample=self.panel.wrench_feedback(self.side) if self.panel.wrench_feedback else None
        violation,*_=evaluate_wrench(sample,self.panel.force_limit.value(),self.panel.torque_limit.value(),
            policy=self.panel.selected_profile().get('force_policy'))
        actor=self.panel.runner or self.panel.hand_runner
        self.pause_required=bool(violation or (actor is not None and actor.state=='paused'))
        self.pause_reason='六维力保护：'+violation if violation else (actor.message if actor is not None and actor.state=='paused' else '')
        if self.vision_error:raise ValueError(self.vision_error)
        if self.vision_pending and time.monotonic()>self.vision_deadline:raise ValueError('真机视觉复核超时')
        if self.action_state!='running' or self.action is None:return
        p=self.panel
        if self.action['kind']=='wait':
            if time.monotonic()-self.started>=self.action.get('duration',1.):self.action_state='complete'
            return
        if p.process is not None or p.runner is not None or p.hand_runner is not None:return
        if p._program_action_result:
            state,message=p._program_action_result;p._program_action_result=None
            if state=='complete':
                self.action_state='complete'
                if self.action['kind']=='hand' and self.action['open']:
                    p.world.detach(self.side,p.feedback());p._unverified_grips.discard(self.side)
                    self.contacts[self.role_for_hand]=False
            else:self.action_state='failed';self.action_error=message
        elif p.sequence.state=='failed':self.action_state='failed';self.action_error=p.message
        elif time.monotonic()-self.started>240:self.action_state='failed';self.action_error='真机动作等待超时'

    def pause(self):
        self.panel._workflow_auto_paused=True
        actor=self.panel.runner or self.panel.hand_runner
        if actor is not None and actor.state=='running':actor.pause()
    def resume(self):
        self.validate();p=self.panel;actor=p.runner or p.hand_runner
        self.tick()
        if self.pause_reason.startswith('六维力保护：'):raise ValueError(self.pause_reason)
        with self.authorized():
            if actor is not None:
                if actor.state!='paused':raise ValueError('尚未确认暂停到位')
                p.pause_or_resume()
            elif p.sequence.state=='planned':p._execute_ready_plan()
        p._workflow_auto_paused=False
    def cancel(self):
        self.cancelled=True;self.vision_generation+=1;self.vision_pending=False
        with self.authorized():self.panel.stop()
        self.action_state='cancelled'
    @property
    def busy(self):return self.panel.process is not None or self.panel.runner is not None or self.panel.hand_runner is not None
